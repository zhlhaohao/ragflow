#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging
import re
from functools import reduce
from io import BytesIO
from timeit import default_timer as timer

from docx import Document
from docx.image.exceptions import InvalidImageStreamError, UnexpectedEndOfFileError, UnrecognizedImageError
from markdown import markdown 
from PIL import Image
from tika import parser

from api.db import LLMType
from api.db.services.llm_service import LLMBundle
from deepdoc.parser import DocxParser, ExcelParser, HtmlParser, JsonParser, MarkdownParser, PdfParser, TxtParser
from deepdoc.parser.figure_parser import VisionFigureParser, vision_figure_parser_figure_data_wraper
from deepdoc.parser.figure_parser import VisionFigureParser, vision_figure_parser_figure_data_wraper
from deepdoc.parser.pdf_parser import PlainParser, VisionParser
from rag.nlp import concat_img, find_codec, naive_merge, naive_merge_with_images, naive_merge_docx, rag_tokenizer, tokenize_chunks, tokenize_chunks_with_images, tokenize_table
from rag.utils import num_tokens_from_string
from api.utils import ic


class Docx(DocxParser):
    """
    解析docx,使用docx库
    """
    def __init__(self):
        pass

    def get_picture(self, document, paragraph):
        """
        从段落中提取图片。它首先检查段落中是否有图片，如果有，则尝试通过图片的嵌入ID获取图片的数据流。如果成功，它会将图片数据转换成 PIL.Image 对象并返回。如果遇到错误（如图片格式不被识别、读取过程中遇到意外结束等），则会打印一条消息并返回 None。
        """
        img = paragraph._element.xpath('.//pic:pic')
        if not img:
            return None
        img = img[0]
        embed = img.xpath('.//a:blip/@r:embed')
        if not embed:
            return None
        embed = embed[0]
        related_part = document.part.related_parts[embed]
        try:
            image_blob = related_part.image.blob
        except UnrecognizedImageError:
            logging.info("Unrecognized image format. Skipping image.")
            return None
        except UnexpectedEndOfFileError:
            logging.info("EOF was unexpectedly encountered while reading an image stream. Skipping image.")
            return None
        except InvalidImageStreamError:
            logging.info("The recognized image stream appears to be corrupted. Skipping image.")
            return None
        except UnicodeDecodeError:
            logging.info("The recognized image stream appears to be corrupted. Skipping image.")
            return None
        try:
            image = Image.open(BytesIO(image_blob)).convert('RGB')
            return image
        except Exception:
            return None

    def __clean(self, line):
        """
        清理段落中的文本。它将全角空格 \u3000 替换为普通的空格，并去除字符串两端的空白字符
        """
        line = re.sub(r"\u3000", " ", line).strip()
        return line

    def __get_nearest_title(self, table_index, filename):
        """Get the hierarchical title structure before the table"""
        import re
        from docx.text.paragraph import Paragraph
        
        titles = []
        blocks = []
        
        # Get document name from filename parameter
        doc_name = re.sub(r"\.[a-zA-Z]+$", "", filename)
        if not doc_name:
            doc_name = "Untitled Document"
            
        # Collect all document blocks while maintaining document order
        try:
            # Iterate through all paragraphs and tables in document order
            for i, block in enumerate(self.doc._element.body):
                if block.tag.endswith('p'):  # Paragraph
                    p = Paragraph(block, self.doc)
                    blocks.append(('p', i, p))
                elif block.tag.endswith('tbl'):  # Table
                    blocks.append(('t', i, None))  # Table object will be retrieved later
        except Exception as e:
            logging.error(f"Error collecting blocks: {e}")
            return ""
            
        # Find the target table position
        target_table_pos = -1
        table_count = 0
        for i, (block_type, pos, _) in enumerate(blocks):
            if block_type == 't':
                if table_count == table_index:
                    target_table_pos = pos
                    break
                table_count += 1
                
        if target_table_pos == -1:
            return ""  # Target table not found
            
        # Find the nearest heading paragraph in reverse order
        nearest_title = None
        for i in range(len(blocks)-1, -1, -1):
            block_type, pos, block = blocks[i]
            if pos >= target_table_pos:  # Skip blocks after the table
                continue
                
            if block_type != 'p':
                continue
                
            if block.style and re.search(r"Heading\s*(\d+)", block.style.name, re.I):
                try:
                    level_match = re.search(r"(\d+)", block.style.name)
                    if level_match:
                        level = int(level_match.group(1))
                        if level <= 7:  # Support up to 7 heading levels
                            title_text = block.text.strip()
                            if title_text:  # Avoid empty titles
                                nearest_title = (level, title_text)
                                break
                except Exception as e:
                    logging.error(f"Error parsing heading level: {e}")
        
        if nearest_title:
            # Add current title
            titles.append(nearest_title)
            current_level = nearest_title[0]
            
            # Find all parent headings, allowing cross-level search
            while current_level > 1:
                found = False
                for i in range(len(blocks)-1, -1, -1):
                    block_type, pos, block = blocks[i]
                    if pos >= target_table_pos:  # Skip blocks after the table
                        continue
                        
                    if block_type != 'p':
                        continue
                        
                    if block.style and re.search(r"Heading\s*(\d+)", block.style.name, re.I):
                        try:
                            level_match = re.search(r"(\d+)", block.style.name)
                            if level_match:
                                level = int(level_match.group(1))
                                # Find any heading with a higher level
                                if level < current_level:  
                                    title_text = block.text.strip()
                                    if title_text:  # Avoid empty titles
                                        titles.append((level, title_text))
                                        current_level = level
                                        found = True
                                        break
                        except Exception as e:
                            logging.error(f"Error parsing parent heading: {e}")
                            
                if not found:  # Break if no parent heading is found
                    break
            
            # Sort by level (ascending, from highest to lowest)
            titles.sort(key=lambda x: x[0])
            # Organize titles (from highest to lowest)
            hierarchy = [doc_name] + [t[1] for t in titles]
            return " > ".join(hierarchy)
            
        return ""

    def __call__(self, filename, binary=None, from_page=0, to_page=100000):
        """
        实现了主要的解析逻辑。它接受文件名或二进制数据作为输入，解析文档的内容，并返回两个列表：
        1. 一个是包含每个段落的文本和段落内部的所有图片拼接在一起的大图片的元组的列表，
        2. 一个是包含表格HTML表示的元组的列表。

        文档中的每个段落都会被处理，如果段落中有文本，则会被清理并存储在 lines 列表中，同时也会尝试从段落中提取图片。
        如果段落样式为 Caption，则会特殊处理，尝试将最近的图片与之关联。
        当遇到页面断点时（由段落中的运行元素中的 lastRenderedPageBreak 或 w:br 标签指示），页码计数器 pn 会增加。
        最后，所有图片列表被合并成单个图片对象（如果有的话），并且所有的表格被转换成HTML格式。
        """


        # 根据是否提供二进制数据来加载文档
        self.doc = Document(
            filename) if not binary else Document(BytesIO(binary))
        
        # 初始化页码计数器和结果列表
        pn = 0
        lines = []
        last_image = None
        
        # 遍历文档中的每个段落
        for p in self.doc.paragraphs:
            if pn > to_page:
                # 如果当前页码超过指定的结束页码，停止处理
                break
            
            if from_page <= pn < to_page:
                # 段落中有文本内容
                if p.text.strip():
                    if p.style and p.style.name == 'Caption':
                        # 特殊处理 Caption 样式的段落
                        former_image = None
                        if lines and lines[-1][1] and lines[-1][2] != 'Caption':
                            # 获取前一个段落的最后一个图片
                            former_image = lines[-1][1].pop()
                        elif last_image:
                            # 使用上一个未处理的图片
                            former_image = last_image
                            last_image = None
                        # 添加 Caption 段落    
                        lines.append((self.__clean(p.text), [former_image], p.style.name))
                    else:
                        # 处理普通段落
                        # 尝试从段落中提取图片
                        current_image = self.get_picture(self.doc, p)
                        image_list = [current_image]
                        if last_image:
                            # 如果有未处理的图片，添加到当前图片列表前面
                            image_list.insert(0, last_image)
                            last_image = None
                        # 添加普通段落
                        lines.append((self.__clean(p.text), image_list, p.style.name if p.style else ""))
                else:
                    # 段落中无文本，但可能有图片
                    if current_image := self.get_picture(self.doc, p):
                        if lines:
                            # 将图片添加到前一个段落的图片列表中
                            lines[-1][1].append(current_image)
                        else:
                            # 如果没有前一个段落，保存图片以备后续处理
                            last_image = current_image
            # 处理段落中的页面断点
            for run in p.runs:
                if 'lastRenderedPageBreak' in run._element.xml:
                    pn += 1
                    continue
                if 'w:br' in run._element.xml and 'type="page"' in run._element.xml:
                    pn += 1

        # 合并每个段落的图片列表,也就是将该段落的多张图片合并为一张大图片，如果没有图片，则为None
        new_line = [(line[0], reduce(concat_img, line[1]) if line[1] else None) for line in lines]

        # 处理文档中的表格
        tbls = []
        for i, tb in enumerate(self.doc.tables):
            title = self.__get_nearest_title(i, filename)
            html = "<table>"
            if title:
                html += f"<caption>Table Location: {title}</caption>"
            for r in tb.rows:
                # 开始新行
                html += "<tr>"
                i = 0
                while i < len(r.cells):
                    span = 1
                    c = r.cells[i]
                    for j in range(i + 1, len(r.cells)):
                        if c.text == r.cells[j].text:
                            # 计算单元格合并的列数
                            span += 1
                            i = j
                        else:
                            break
                    i += 1
                    # 添加单元格
                    html += f"<td>{c.text}</td>" if span == 1 else f"<td colspan='{span}'>{c.text}</td>"
                # 结束行    
                html += "</tr>"
            # 结束表格    
            html += "</table>"
            # 将表格的 HTML 表示添加到结果列表中
            tbls.append(((None, html), ""))

        # 返回处理后的文本和图片列表，以及表格列表
        return new_line, tbls

class Pdf(PdfParser):
    def __init__(self):
        super().__init__()

    def __call__(self, filename, binary=None, from_page=0,
                 to_page=100000, zoomin=3, callback=None, separate_tables_figures=False):
        start = timer()
        first_start = start
        callback(msg="OCR开启")
        self.__images__(
            filename if not binary else binary,
            zoomin,
            from_page,
            to_page,
            callback
        )
        callback(msg="OCR结束 ({:.2f}s)".format(timer() - start))
        logging.info("OCR({}~{}): {:.2f}s".format(from_page, to_page, timer() - start))

        start = timer()
        self._layouts_rec(zoomin)
        callback(0.63, "分析布局 ({:.2f}s)".format(timer() - start))

        start = timer()
        self._table_transformer_job(zoomin)
        callback(0.65, "分析表格 ({:.2f}s)".format(timer() - start))

        start = timer()
        self._text_merge()
        callback(0.67, "Text merged ({:.2f}s)".format(timer() - start))

        if separate_tables_figures:
            tbls, figures = self._extract_table_figure(True, zoomin, True, True, True)
            self._concat_downward()
            logging.info("layouts cost: {}s".format(timer() - first_start))
            return [(b["text"], self._line_tag(b, zoomin)) for b in self.boxes], tbls, figures
        else:
            tbls = self._extract_table_figure(True, zoomin, True, True)
            # self._naive_vertical_merge()
            self._concat_downward()
            # self._filter_forpages()
            logging.info("layouts cost: {}s".format(timer() - first_start))
            return [(b["text"], self._line_tag(b, zoomin)) for b in self.boxes], tbls


class Markdown(MarkdownParser):
    def get_picture_urls(self, sections):
        if not sections:
            return []
        if isinstance(sections, type("")):
            text = sections
        elif isinstance(sections[0], type("")):
            text = sections[0]
        else:
            return []
        
        from bs4 import BeautifulSoup
        html_content = markdown(text)
        soup = BeautifulSoup(html_content, 'html.parser')
        html_images = [img.get('src') for img in soup.find_all('img') if img.get('src')]
        return html_images
    
    def get_pictures(self, text):
        """Download and open all images from markdown text."""
        import requests
        image_urls = self.get_picture_urls(text)
        images = []
        # Find all image URLs in text
        for url in image_urls:
            try:
                response = requests.get(url, stream=True, timeout=30)
                if response.status_code == 200 and response.headers['Content-Type'].startswith('image/'):
                    img = Image.open(BytesIO(response.content)).convert('RGB')
                    images.append(img)
            except Exception as e:
                logging.error(f"Failed to download/open image from {url}: {e}")
                continue
                    
        return images if images else None

    def __call__(self, filename, binary=None):
        if binary:
            encoding = find_codec(binary)
            txt = binary.decode(encoding, errors="ignore")
        else:
            with open(filename, "r") as f:
                txt = f.read()
        remainder, tables = self.extract_tables_and_remainder(f'{txt}\n')
        sections = []
        tbls = []
        for sec in remainder.split("\n"):
            if num_tokens_from_string(sec) > 3 * self.chunk_token_num:
                sections.append((sec[:int(len(sec) / 2)], ""))
                sections.append((sec[int(len(sec) / 2):], ""))
            else:
                if sec.strip().find("#") == 0:
                    sections.append((sec, ""))
                elif sections and sections[-1][0].strip().find("#") == 0:
                    sec_, _ = sections.pop(-1)
                    sections.append((sec_ + "\n" + sec, ""))
                else:
                    sections.append((sec, ""))
        for table in tables:
            tbls.append(((None, markdown(table, extensions=['markdown.extensions.tables'])), ""))
        return sections, tbls


def chunk(filename, binary=None, from_page=0, to_page=100000,
          lang="Chinese", callback=None, **kwargs):
    """
        Supported file formats are docx, pdf, excel, txt.
        This method apply the naive ways to chunk files.
        Successive text will be sliced into pieces using 'delimiter'.
        Next, these successive pieces are merge into chunks whose token number is no more than 'Max token number'.
    """

    is_english = lang.lower() == "english"  # is_english(cks)
    parser_config = kwargs.get(
        "parser_config", {
            "chunk_token_num": 128, "delimiter": "\n!?。；！？", "layout_recognize": "DeepDOC"})
    doc = {
        "docnm_kwd": filename,
        "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename))
    }
    doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])
    res = []
    pdf_parser = None
    section_images = None
    if re.search(r"\.docx$", filename, re.IGNORECASE):
        callback(0.1, "开始解析.")

        try:
            vision_model = LLMBundle(kwargs["tenant_id"], LLMType.IMAGE2TEXT)
            callback(0.15, "Visual model detected. Attempting to enhance figure extraction...")
        except Exception:
            vision_model = None

        sections, tables = Docx()(filename, binary)

        if vision_model:
            figures_data = vision_figure_parser_figure_data_wraper(sections)
            try:
                docx_vision_parser = VisionFigureParser(vision_model=vision_model, figures_data=figures_data, **kwargs)
                boosted_figures = docx_vision_parser(callback=callback)
                tables.extend(boosted_figures)
            except Exception as e:
                callback(0.6, f"Visual model error: {e}. Skipping figure parsing enhancement.")

        res = tokenize_table(tables, doc, is_english)
        callback(0.8, "Finish parsing.")

        st = timer()

        # 将段落合并为chunks
        chunks, images = naive_merge_docx(
            sections, int(parser_config.get(
                "chunk_token_num", 128)), parser_config.get(
                "delimiter", "\n!?。；！？"))

        # section_only=True 直接返回段落,但是不会发生
        if kwargs.get("section_only", False):
            return chunks

        res.extend(tokenize_chunks_with_images(chunks, doc, is_english, images))
        logging.info("naive_merge({}): {}".format(filename, timer() - st))
        return res

    # 如果是pdf文件
    elif re.search(r"\.pdf$", filename, re.IGNORECASE):
        layout_recognizer = parser_config.get("layout_recognize", "DeepDOC")
        if isinstance(layout_recognizer, bool):
            layout_recognizer = "DeepDOC" if layout_recognizer else "Plain Text"
        callback(0.1, "Start to parse.")

        if layout_recognizer == "DeepDOC":
            pdf_parser = Pdf()

            try:
                vision_model = LLMBundle(kwargs["tenant_id"], LLMType.IMAGE2TEXT)
                callback(0.15, "Visual model detected. Attempting to enhance figure extraction...")
            except Exception:
                vision_model = None

            if vision_model:
                sections, tables, figures = pdf_parser(filename if not binary else binary, from_page=from_page, to_page=to_page, callback=callback, separate_tables_figures=True)
                callback(0.5, "Basic parsing complete. Proceeding with figure enhancement...")
                try:
                    pdf_vision_parser = VisionFigureParser(vision_model=vision_model, figures_data=figures, **kwargs)
                    boosted_figures = pdf_vision_parser(callback=callback)
                    tables.extend(boosted_figures)
                except Exception as e:
                    callback(0.6, f"Visual model error: {e}. Skipping figure parsing enhancement.")
                    tables.extend(figures)
            else:
                sections, tables = pdf_parser(filename if not binary else binary, from_page=from_page, to_page=to_page, callback=callback)

            res = tokenize_table(tables, doc, is_english)
            callback(0.8, "Finish parsing.")

        else:
            if layout_recognizer == "Plain Text":
                pdf_parser = PlainParser()
            else:
                vision_model = LLMBundle(kwargs["tenant_id"], LLMType.IMAGE2TEXT, llm_name=layout_recognizer, lang=lang)
                pdf_parser = VisionParser(vision_model=vision_model, **kwargs)

            sections, tables = pdf_parser(filename if not binary else binary, from_page=from_page, to_page=to_page,
                                          callback=callback)
            res = tokenize_table(tables, doc, is_english)
            callback(0.8, "Finish parsing.")

    elif re.search(r"\.(csv|xlsx?)$", filename, re.IGNORECASE):
        callback(0.1, "Start to parse.")
        excel_parser = ExcelParser()
        # 检查解析配置中是否启用了"html4excel"选项
        if parser_config.get("html4excel"):
            # 如果启用，则调用excel_parser的html方法解析文件，并将结果转换为(section, "")格式的列表
            sections = [(_, "") for _ in excel_parser.html(binary, 12) if _]
        else:
            # 如果未启用，则直接调用excel_parser的方法解析文件，并将结果转换为(section, "")格式的列表
            sections = [(_, "") for _ in excel_parser(binary) if _]

    elif re.search(r"\.(txt|py|js|java|c|cpp|h|php|go|ts|sh|cs|kt|sql)$", filename, re.IGNORECASE):
        callback(0.1, "开始解析.")
        sections = TxtParser()(filename, binary,
                               parser_config.get("chunk_token_num", 128),
                               parser_config.get("delimiter", "\n!?;。；！？"))
        callback(0.8, "Finish parsing.")

    elif re.search(r"\.(md|markdown)$", filename, re.IGNORECASE):
        callback(0.1, "开始解析.")
        markdown_parser = Markdown(int(parser_config.get("chunk_token_num", 128)))
        sections, tables = markdown_parser(filename, binary)
        
        # Process images for each section
        section_images = []
        for section_text, _ in sections:
            images = markdown_parser.get_pictures(section_text) if section_text else None
            if images:
                # If multiple images found, combine them using concat_img
                combined_image = reduce(concat_img, images) if len(images) > 1 else images[0]
                section_images.append(combined_image)
            else:
                section_images.append(None)
                
        res = tokenize_table(tables, doc, is_english)
        callback(0.8, "完成解析.")

    elif re.search(r"\.(htm|html)$", filename, re.IGNORECASE):
        callback(0.1, "开始解析.")
        sections = HtmlParser()(filename, binary)
        sections = [(_, "") for _ in sections if _]
        callback(0.8, "Finish parsing.")

    elif re.search(r"\.json$", filename, re.IGNORECASE):
        callback(0.1, "开始解析.")
        chunk_token_num = int(parser_config.get("chunk_token_num", 128))
        sections = JsonParser(chunk_token_num)(binary)
        sections = [(_, "") for _ in sections if _]
        callback(0.8, "Finish parsing.")

    elif re.search(r"\.doc$", filename, re.IGNORECASE):
        callback(0.1, "开始解析.")
        binary = BytesIO(binary)
        doc_parsed = parser.from_buffer(binary)
        if doc_parsed.get('content', None) is not None:
            sections = doc_parsed['content'].split('\n')
            sections = [(_, "") for _ in sections if _]
            callback(0.8, "Finish parsing.")
        else:
            callback(0.8, f"tika.parser got empty content from {filename}.")
            logging.warning(f"tika.parser got empty content from {filename}.")
            return []

    else:
        raise NotImplementedError(
            "file type not supported yet(pdf, xlsx, doc, docx, txt supported)")

    st = timer()
    if section_images:
        # if all images are None, set section_images to None
        if all(image is None for image in section_images):
            section_images = None

    if section_images:
        chunks, images = naive_merge_with_images(sections, section_images,
                                        int(parser_config.get(
                                            "chunk_token_num", 128)), parser_config.get(
                                            "delimiter", "\n!?。；！？"))
        if kwargs.get("section_only", False):
            return chunks
        
        res.extend(tokenize_chunks_with_images(chunks, doc, is_english, images))
    else:
        chunks = naive_merge(
            sections, int(parser_config.get(
                "chunk_token_num", 128)), parser_config.get(
                "delimiter", "\n!?。；！？"))
        if kwargs.get("section_only", False):
            return chunks

    # 分词，变成了 content_with_weight： 原文 content_ltks：分词，用空格分开单词 content_sm_ltks：细粒度分词
    res.extend(tokenize_chunks(chunks, doc, is_english, pdf_parser))
    logging.info("naive_merge({}): {}".format(filename, timer() - st))
    return res


if __name__ == "__main__":
    import sys

    def dummy(prog=None, msg=""):
        pass

    chunk(sys.argv[1], from_page=0, to_page=10, callback=dummy)
