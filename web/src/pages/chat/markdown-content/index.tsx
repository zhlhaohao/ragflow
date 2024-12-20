import Image from '@/components/image';
import SvgIcon from '@/components/svg-icon';
import { IReference } from '@/interfaces/database/chat';
import { IChunk } from '@/interfaces/database/knowledge';
import { getExtension } from '@/utils/document-util';
import { InfoCircleOutlined } from '@ant-design/icons';
import { Button, Flex, Popover, Space } from 'antd';
import DOMPurify from 'dompurify';
import { useCallback, useEffect, useMemo } from 'react';
import Markdown from 'react-markdown';
import reactStringReplace from 'react-string-replace';
import SyntaxHighlighter from 'react-syntax-highlighter';
import remarkGfm from 'remark-gfm';
import { visitParents } from 'unist-util-visit-parents';

import { useFetchDocumentThumbnailsByIds } from '@/hooks/document-hooks';
import { useTranslation } from 'react-i18next';
import styles from './index.less';

// reg 将匹配形如 ##1$$, ##234$$ 的字符串
const reg = /(#{2}\d+\${2})/g;
// curReg 将匹配形如 ~~1$$, ~~234$$ 的字符串
const curReg = /(~{2}\d+\${2})/g;

const getChunkIndex = (match: string) => Number(match.slice(2, -2));
// TODO: The display of the table is inconsistent with the display previously placed in the MessageItem.
/* 
MarkdownContent 是一个自定义的 React 组件，旨在渲染 Markdown 格式的内容，并根据特定的模式在内容中嵌入交互元素
当 Markdown 组件接收到 contentWithCursor 作为其内容时，它会根据配置好的插件和组件规则解析并渲染这段 Markdown 文本。
对于任何匹配到的 ##数字$$ 或 ~~数字$$ 模式，renderReference 会生成相应的 Popover 组件或者其他 UI 元素。
如果文本中包含了代码块，并且指定了编程语言，则 SyntaxHighlighter 会被用来提供漂亮的代码高亮效果。
 */
const MarkdownContent = ({
  reference,
  clickDocumentButton,
  content,
  loading,
}: {
  content: string;
  loading: boolean;
  reference: IReference;
  clickDocumentButton?: (documentId: string, chunk: IChunk) => void;
}) => {
  const { t } = useTranslation();
  const { setDocumentIds, data: fileThumbnails } =
    useFetchDocumentThumbnailsByIds();

  // 提供一种机制，可以在内容加载过程中向用户反馈当前状态，而不会影响到原始的 Markdown 内容。通过这种方式，即使内容尚未完全准备好，用户也能得到即时的视觉反馈，知道系统正在工作
  const contentWithCursor = useMemo(() => {
    let text = content;
    // 如果 text 是空字符串，并且组件处于非加载状态，那么它会用国际化后的 "搜索中" 提示信息替换 text。这里使用了 t 函数来进行多语言支持，确保提示信息可以按照用户的语言设置正确显示。
    if (text === '') {
      text = t('chat.searching');
    }

    // 加载状态下的特殊标记：如果 loading 状态为真（即正在进行某些异步操作），则会在 text 的末尾追加一个特殊的标记 '~~2$$'。这个标记可能是为了指示光标位置或者用于其他特定目的，比如在用户界面中显示加载进度或占位符。
    return loading ? text?.concat('~~2$$') : text;
  }, [content, loading, t]);

  /* 
    这段 useEffect 的主要目的是确保每次 reference 发生变化时，都会根据最新的 reference.doc_aggs 提取所有相关的文档 ID，并通过 setDocumentIds 更新到组件的状态或其他地方。这样做可以保证组件始终拥有最新且正确的文档 ID 列表，以便进一步处理，例如加载文档缩略图等。
    此外，将 setDocumentIds 包含在依赖项数组中是为了防止潜在的闭包问题，确保每次调用时都使用的是最新的 setDocumentIds 函数实例。不过，通常情况下，setDocumentIds 是由 useState 或自定义 hook 返回的 setter 函数，它不会改变，因此将其包含在依赖项中并不是严格必要的。如果你确定 setDocumentIds 不会变化，可以考虑将其从依赖项数组中移除以优化性能。
  */
  useEffect(() => {
    setDocumentIds(reference?.doc_aggs?.map((x) => x.doc_id) ?? []);
  }, [reference, setDocumentIds]);

  // 当点击文档按钮时触发的回调函数。它检查文件是否为 PDF，如果是，则调用父组件传递的方法 `clickDocumentButton`。
  const handleDocumentButtonClick = useCallback(
    (documentId: string, chunk: IChunk, isPdf: boolean) => () => {
      if (!isPdf) {
        return;
      }
      clickDocumentButton?.(documentId, chunk);
    },
    [clickDocumentButton],
  );

  // 当 Markdown 内容中有匹配到 ##1$$, ##234$$ 的字符串时，rehypePlugins使用来修改AST树。
  // 它遍历所有文本节点，并将它们包装在 <custom-typography> 元素中，除非这些文本已经位于 <custom-typography> 或 <code> 标签内。
  const rehypeWrapReference = () => {
    return function wrapTextTransform(tree: any) {
      visitParents(tree, 'text', (node, ancestors) => {
        const latestAncestor = ancestors.at(-1);
        if (
          latestAncestor.tagName !== 'custom-typography' &&
          latestAncestor.tagName !== 'code'
        ) {
          node.type = 'element';
          node.tagName = 'custom-typography';
          node.properties = {};
          node.children = [{ type: 'text', value: node.value }];
        }
      });
    };
  };

  // 获取弹出窗口的内容，当用户悬停在信息图标上时显示。它根据 chunkIndex 来确定要显示的具体内容。
  const getPopoverContent = useCallback(
    (chunkIndex: number) => {
      const chunks = reference?.chunks ?? [];
      const chunkItem = chunks[chunkIndex];
      const document = reference?.doc_aggs?.find(
        (x) => x?.doc_id === chunkItem?.doc_id,
      );
      const documentId = document?.doc_id;
      const fileThumbnail = documentId ? fileThumbnails[documentId] : '';
      const fileExtension = documentId ? getExtension(document?.doc_name) : '';
      const imageId = chunkItem?.img_id;
      return (
        <Flex
          key={chunkItem?.chunk_id}
          gap={10}
          className={styles.referencePopoverWrapper}
        >
          {imageId && (
            <Popover
              placement="left"
              content={
                <Image
                  id={imageId}
                  className={styles.referenceImagePreview}
                ></Image>
              }
            >
              <Image
                id={imageId}
                className={styles.referenceChunkImage}
              ></Image>
            </Popover>
          )}
          <Space direction={'vertical'}>
            <div
              dangerouslySetInnerHTML={{
                __html: DOMPurify.sanitize(chunkItem?.content_with_weight),
              }}
              className={styles.chunkContentText}
            ></div>
            {documentId && (
              <Flex gap={'small'}>
                {fileThumbnail ? (
                  <img
                    src={fileThumbnail}
                    alt=""
                    className={styles.fileThumbnail}
                  />
                ) : (
                  <SvgIcon
                    name={`file-icon/${fileExtension}`}
                    width={24}
                  ></SvgIcon>
                )}
                <Button
                  type="link"
                  className={styles.documentLink}
                  onClick={handleDocumentButtonClick(
                    documentId,
                    chunkItem,
                    fileExtension === 'pdf',
                  )}
                >
                  {document?.doc_name}
                </Button>
              </Flex>
            )}
          </Space>
        </Flex>
      );
    },
    [reference, fileThumbnails, handleDocumentButtonClick],
  );

  // 渲染引用标记（也就是将##1$$渲染为(i)）的回调函数。它会替换特定模式的文本为 Popover 组件或带样式类的 span 元素。
  const renderReference = useCallback(
    (text: string) => {
      let replacedText = reactStringReplace(text, reg, (match, i) => {
        const chunkIndex = getChunkIndex(match);
        return (
          <Popover content={getPopoverContent(chunkIndex)} key={i}>
            <InfoCircleOutlined className={styles.referenceIcon} />
          </Popover>
        );
      });

      replacedText = reactStringReplace(replacedText, curReg, (match, i) => (
        <span className={styles.cursor} key={i}></span>
      ));

      return replacedText;
    },
    [getPopoverContent],
  );

  /* 
    最后返回最终的Markdown：
    1. remark-gfm：插件扩展了标准 Markdown 语法，支持 GitHub 风格的 Markdown 特性，如表格、任务列表等。
    2. components：允许你覆盖默认的 HTML 元素渲染方式。你可以指定特定的 React 组件来代替原生 HTML 标签。
    3. custom-typography: 定义了一个自定义组件，当遇到 <custom-typography> 标签时会调用这个组件。这里实际上是指向 renderReference 函数，该函数负责处理引用标记，并将它们替换为 Popover 或者带有样式类的 span 元素。
    4. code: 自定义了 <code> 标签的渲染方式。如果 <code> 标签包含语言信息（例如 language-js），则使用 SyntaxHighlighter 组件进行代码高亮显示；否则，按常规方式渲染代码块。
    5. contentWithCursor：这是传递给 <Markdown> 组件的内容，即经过处理的 Markdown 文本。contentWithCursor 包含原始的 Markdown 内容以及可能附加的加载指示符。
  */
  return (
    <Markdown
      rehypePlugins={[rehypeWrapReference]}
      remarkPlugins={[remarkGfm]}
      components={
        {
          'custom-typography': ({ children }: { children: string }) =>
            renderReference(children),
          code(props: any) {
            const { children, className, node, ...rest } = props;
            const match = /language-(\w+)/.exec(className || '');
            return match ? (
              <SyntaxHighlighter {...rest} PreTag="div" language={match[1]}>
                {String(children).replace(/\n$/, '')}
              </SyntaxHighlighter>
            ) : (
              <code {...rest} className={className}>
                {children}
              </code>
            );
          },
        } as any
      }
    >
      {contentWithCursor}
    </Markdown>
  );
};

export default MarkdownContent;
