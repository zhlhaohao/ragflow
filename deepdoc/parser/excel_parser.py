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
import sys
from io import BytesIO

import pandas as pd
from openpyxl import Workbook, load_workbook

from rag.nlp import find_codec
from api.utils import ic


class RAGFlowExcelParser:

    @staticmethod
    def _load_excel_to_workbook(file_like_object):
        if isinstance(file_like_object, bytes):
            file_like_object = BytesIO(file_like_object)

        # Read first 4 bytes to determine file type
        file_like_object.seek(0)
        file_head = file_like_object.read(4)
        file_like_object.seek(0)

        if not (file_head.startswith(b'PK\x03\x04') or file_head.startswith(b'\xD0\xCF\x11\xE0')):
            logging.info("****wxy: Not an Excel file, converting CSV to Excel Workbook")

            try:
                file_like_object.seek(0)
                df = pd.read_csv(file_like_object)
                return RAGFlowExcelParser._dataframe_to_workbook(df)

            except Exception as e_csv:
                raise Exception(f"****wxy: Failed to parse CSV and convert to Excel Workbook: {e_csv}")

        try:
            return load_workbook(file_like_object, data_only=True)
        except Exception as e:
            logging.info(f"****wxy: openpyxl load error: {e}, try pandas instead")
            try:
                file_like_object.seek(0)
                df = pd.read_excel(file_like_object)
                return RAGFlowExcelParser._dataframe_to_workbook(df)
            except Exception as e_pandas:
                raise Exception(f"****wxy: pandas.read_excel error: {e_pandas}, original openpyxl error: {e}")

    @staticmethod
    def _dataframe_to_workbook(df):
        wb = Workbook()
        ws = wb.active
        ws.title = "Data"

        for col_num, column_name in enumerate(df.columns, 1):
            ws.cell(row=1, column=col_num, value=column_name)

        for row_num, row in enumerate(df.values, 2):
            for col_num, value in enumerate(row, 1):
                ws.cell(row=row_num, column=col_num, value=value)

        return wb

    def html(self, fnm, chunk_rows=256):
        file_like_object = BytesIO(fnm) if not isinstance(fnm, str) else fnm
        wb = RAGFlowExcelParser._load_excel_to_workbook(file_like_object)
        tb_chunks = []
        # 遍历Excel文件中的每个工作表
        for sheetname in wb.sheetnames:
            ws = wb[sheetname]
            rows = list(ws.rows)
            if not rows:
                continue
            # F8080 检测表头行
            header_row_index = self.detect_header_row(rows)
            # 构建表头行的HTML字符串
            tb_rows_0 = "<tr>"
            for t in list(rows[header_row_index]):
                tb_rows_0 += f"<th>{t.value}</th>"
            tb_rows_0 += "</tr>"
            # 将数据分块，每块包含chunk_rows行（不包括表头）
            for chunk_i in range((len(rows) - header_row_index - 1) // chunk_rows + 1):
                tb = ""
                tb += f"<table><caption>{sheetname}</caption>"
                tb += tb_rows_0
                for r in list(
                    rows[header_row_index + 1 + chunk_i * chunk_rows : header_row_index + 1 + (chunk_i + 1) * chunk_rows]
                ):
                    tb += "<tr>"
                    for i, c in enumerate(r):
                        if c.value is None:
                            tb += "<td></td>"
                        else:
                            tb += f"<td>{c.value}</td>"
                    tb += "</tr>"
                tb += "</table>\n"
                tb_chunks.append(tb)
        return tb_chunks

    def __call__(self, fnm):
        file_like_object = BytesIO(fnm) if not isinstance(fnm, str) else fnm
        wb = RAGFlowExcelParser._load_excel_to_workbook(file_like_object)

        res = []
        # 遍历Excel文件中的每个工作表
        for sheetname in wb.sheetnames:
            ws = wb[sheetname]
            rows = list(ws.rows)
            if not rows:
                continue
            # F8080 检测表头行
            header_row_index = self.detect_header_row(rows)
            ti = list(rows[header_row_index])
            # 遍历数据行（不包括表头）
            for r in list(rows[header_row_index + 1:]):
                # 用于存储当前行的字段
                fields = []
                # 遍历当前行的单元格，c代表单元格,i代表单元格序号
                for i, c in enumerate(r):
                    # ic(c.value)
                    cell_value = c.value
                    if not cell_value:
                        # F8080 检查是否为合并单元格
                        for merged_range in ws.merged_cells.ranges:
                            if c.coordinate in merged_range:
                                # 获取合并单元格的值
                                cell_value = ws[merged_range.start_cell.coordinate].value
                                break
                    if not cell_value:
                        continue
                    # 获取表头单元格的值
                    t = str(ti[i].value) if i < len(ti) else ""
                    # 添加当前单元格的值
                    t += ("：" if t else "") + str(cell_value)
                    # ic(t)
                    # 将字段添加到列表中
                    fields.append(t)
                  # 将字段用分号连接成字符串
                line = "; ".join(fields)
                # 如果工作表名不包含"sheet"，则添加工作表名
                if sheetname.lower().find("sheet") < 0:
                    line += " ——" + sheetname
                # ic(line)
                # 将结果字符串添加到列表中
                res.append(line)
        # ic(res)
        return res

    @staticmethod
    def row_number(fnm, binary):
        """
        计算Excel文件中的总行数。

        参数:
        fnm (str): 文件名。
        binary (bytes): 文件的二进制数据。

        返回:
        int: 文件中的总行数。
        """
        if fnm.split(".")[-1].lower().find("xls") >= 0:
            wb = RAGFlowExcelParser._load_excel_to_workbook(BytesIO(binary))
            total = 0
            for sheetname in wb.sheetnames:
                ws = wb[sheetname]
                total += len(list(ws.rows))
            return total

        if fnm.split(".")[-1].lower() in ["csv", "txt"]:
            encoding = find_codec(binary)
            txt = binary.decode(encoding, errors="ignore")
            return len(txt.split("\n"))

    def detect_header_row(self, rows):
        """
        F8080 检测表头在哪一行。如果 N 行非空单元格数量小于 N+1 行非空单元格数量，则 N+1 为表头行。

        参数:
        rows (list): Excel工作表的所有行。

        返回:
        int: 表头行的索引。
        """
        for i in range(len(rows) - 1):
            row1 = [cell.value for cell in rows[i] if cell.value is not None]
            row2 = [cell.value for cell in rows[i + 1] if cell.value is not None]
            if len(row1) < len(row2) and i < 3:
                return i+1
        return 0

if __name__ == "__main__":
    psr = RAGFlowExcelParser()
    psr(sys.argv[1])
