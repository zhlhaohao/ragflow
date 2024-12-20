import jsPreviewExcel from '@js-preview/excel';
import axios from 'axios';
import mammoth from 'mammoth';
import { useCallback, useEffect, useRef, useState } from 'react';

export const useCatchError = (api: string) => {
  const [error, setError] = useState('');
  const fetchDocument = useCallback(async () => {
    const ret = await axios.get(api);
    const { data } = ret;
    if (!(data instanceof ArrayBuffer) && data.retcode !== 0) {
      setError(data.retmsg);
    }
    return ret;
  }, [api]);

  useEffect(() => {
    fetchDocument();
  }, [fetchDocument]);

  return { fetchDocument, error };
};

// 根据doc_id读文件内容
export const useFetchDocument = () => {
  const fetchDocument = useCallback(async (api: string) => {
    // api = /v1/document/get/a6fa3a5ab85c11ef8d3700155d749c8b
    console.log('[ api ]-26', api);
    const ret = await axios.get(api, { responseType: 'arraybuffer' });
    return ret;
  }, []);

  return { fetchDocument };
};

// 读取excel并渲染
export const useFetchExcel = (filePath: string) => {
  const [status, setStatus] = useState(true);
  const { fetchDocument } = useFetchDocument();
  const containerRef = useRef<HTMLDivElement>(null);
  const { error } = useCatchError(filePath);

  const fetchDocumentAsync = useCallback(async () => {
    let myExcelPreviewer;
    if (containerRef.current) {
      myExcelPreviewer = jsPreviewExcel.init(containerRef.current);
    }
    // 读取文件内容
    const jsonFile = await fetchDocument(filePath);

    // myExcelPreviewer.preview来渲染excel
    myExcelPreviewer
      ?.preview(jsonFile.data)
      .then(() => {
        console.log('succeed');
        setStatus(true);
      })
      .catch((e) => {
        console.warn('failed', e);
        myExcelPreviewer.destroy();
        setStatus(false);
      });
  }, [filePath, fetchDocument]);

  // 在组件mount后，自动执行fetchDocumentAsync
  useEffect(() => {
    fetchDocumentAsync();
  }, [fetchDocumentAsync]);

  return { status, containerRef, error };
};

export const useFetchDocx = (filePath: string) => {
  const [succeed, setSucceed] = useState(true);
  const { fetchDocument } = useFetchDocument();
  const containerRef = useRef<HTMLDivElement>(null);
  const { error } = useCatchError(filePath);

  const fetchDocumentAsync = useCallback(async () => {
    const jsonFile = await fetchDocument(filePath);
    mammoth
      .convertToHtml(
        { arrayBuffer: jsonFile.data },
        { includeDefaultStyleMap: true },
      )
      .then((result) => {
        setSucceed(true);
        const docEl = document.createElement('div');
        docEl.className = 'document-container';
        docEl.innerHTML = result.value;
        const container = containerRef.current;
        if (container) {
          container.innerHTML = docEl.outerHTML;
        }
      })
      .catch(() => {
        setSucceed(false);
      });
  }, [filePath, fetchDocument]);

  useEffect(() => {
    fetchDocumentAsync();
  }, [fetchDocumentAsync]);

  return { succeed, containerRef, error };
};
