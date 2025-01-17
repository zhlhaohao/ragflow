#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
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
from concurrent.futures import ThreadPoolExecutor, ALL_COMPLETED, wait
from threading import Lock
import umap
import numpy as np
from sklearn.mixture import GaussianMixture

from graphrag.utils import get_llm_cache, get_embed_cache, set_embed_cache, set_llm_cache
from rag.utils import truncate


class RecursiveAbstractiveProcessing4TreeOrganizedRetrieval:
    def __init__(self, max_cluster, llm_model, embd_model, prompt, max_token=512, threshold=0.1):
        """
        初始化类实例。

        :param max_cluster: 最大聚类数量。
        :param llm_model: 语言模型，用于生成摘要。
        :param embd_model: 嵌入模型，用于获取文本嵌入。
        :param prompt: 用于生成摘要的提示模板。
        :param max_token: 最大摘要长度，默认为256。
        :param threshold: 聚类阈值，默认为0.1。
        """
        self._max_cluster = max_cluster
        self._llm_model = llm_model
        self._embd_model = embd_model
        self._threshold = threshold
        self._prompt = prompt
        self._max_token = max_token

    def _chat(self, system, history, gen_conf):
        response = get_llm_cache(self._llm_model.llm_name, system, history, gen_conf)
        if response:
            return response
        response = self._llm_model.chat(system, history, gen_conf)
        if response.find("**ERROR**") >= 0:
            raise Exception(response)
        set_llm_cache(self._llm_model.llm_name, system, response, history, gen_conf)
        return response

    def _embedding_encode(self, txt):
        response = get_embed_cache(self._embd_model.llm_name, txt)
        if response:
            return response
        embds, _ = self._embd_model.encode([txt])
        if len(embds) < 1 or len(embds[0]) < 1:
            raise Exception("Embedding error: ")
        embds = embds[0]
        set_embed_cache(self._embd_model.llm_name, txt, embds)
        return embds

    def _get_optimal_clusters(self, embeddings: np.ndarray, random_state: int):
        max_clusters = min(self._max_cluster, len(embeddings))
        n_clusters = np.arange(1, max_clusters)
        bics = []
        for n in n_clusters:
            gm = GaussianMixture(n_components=n, random_state=random_state)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
        optimal_clusters = n_clusters[np.argmin(bics)]
        return optimal_clusters

    def __call__(self, chunks, random_state, callback=None):
        """
        对文档切片进行层次化的聚类和摘要生成。

        :param chunks: 包含文档片段及其嵌入的元组列表。
        :param random_state: 随机状态种子，用于确保结果的可复现性。
        :param callback: 回调函数，用于报告进度。
        """
        layers = [(0, len(chunks))]
        # 初始化开始和结束索引
        start, end = 0, len(chunks)
        # 如果只有一个或没有文档切片，直接返回
        if len(chunks) <= 1:
            return  

        # 过滤掉嵌入为空的文档切片
        chunks = [(s, a) for s, a in chunks if len(a) > 0]

        def summarize(ck_idx, lock):
            """
            对指定索引的文档切片生成摘要,对摘要进行嵌入，然后将(摘要,嵌入)附加到chunks数组。

            :param ck_idx: 文档切片的索引列表。
            :param lock: 线程锁，用于同步访问chunks列表。
            """
            nonlocal chunks
            try:
                texts = [chunks[i][0] for i in ck_idx]
                len_per_chunk = int((self._llm_model.max_length - self._max_token)/len(texts))
                # 截断文本以适应最大长度限制
                cluster_content = "\n".join([truncate(t, max(1, len_per_chunk)) for t in texts])
                # 使用语言模型生成摘要
                cnt = self._chat("You're a helpful assistant.",
                                           [{"role": "user",
                                             "content": self._prompt.format(cluster_content=cluster_content)}],
                                           {"temperature": 0.3, "max_tokens": self._max_token}
                                           )
                cnt = re.sub("(······\n由于长度的原因，回答被截断了，要继续吗？|For the content length reason, it stopped, continue?)", "",
                             cnt)
                logging.debug(f"SUM: {cnt}")
                embds, _ = self._embd_model.encode([cnt])
                with lock:
                    # 将摘要及其嵌入追加到chunks列表中
                    chunks.append((cnt, self._embedding_encode(cnt)))
            except Exception as e:
                logging.exception("summarize got exception")
                return e

        # ---- end of summarize

        # 计算有多少个簇（聚类）
        # 初始化标签列表
        labels = []  
        while end - start > 1:
            # 获取当前层的嵌入
            embeddings = [embd for _, embd in chunks[start:end]]  
            if len(embeddings) == 2:
                # 如果当前层只有两个嵌入，则直接生成摘要
                summarize([start, start + 1], Lock())
                if callback:
                    callback(msg="Cluster one layer: {} -> {}".format(end - start, len(chunks) - end))
                labels.extend([0, 0])
                layers.append((end, len(chunks)))
                start = end
                end = len(chunks)
                continue

            # 计算邻居数量    
            n_neighbors = int((len(embeddings) - 1) ** 0.8)
            # 对嵌入进行降维
            reduced_embeddings = umap.UMAP(
                n_neighbors=max(2, n_neighbors), n_components=min(12, len(embeddings) - 2), metric="cosine"
            ).fit_transform(embeddings)
            # 选择最优的聚类数量  
            n_clusters = self._get_optimal_clusters(reduced_embeddings, random_state)
            if n_clusters == 1:
                # 如果只有一个聚类，则所有嵌入属于同一个簇
                lbls = [0 for _ in range(len(reduced_embeddings))]
            else:
                gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
                gm.fit(reduced_embeddings)
                probs = gm.predict_proba(reduced_embeddings)
                # 根据概率分配标签
                lbls = [np.where(prob > self._threshold)[0] for prob in probs]
                lbls = [lbl[0] if isinstance(lbl, np.ndarray) else lbl for lbl in lbls]
            # 创建线程锁
            lock = Lock()

            # 对每个簇生成摘要
            with ThreadPoolExecutor(max_workers=12) as executor:
                threads = []
                for c in range(n_clusters):
                    ck_idx = [i + start for i in range(len(lbls)) if lbls[i] == c]
                    threads.append(executor.submit(summarize, ck_idx, lock))
                # 等待所有任务完成    
                wait(threads, return_when=ALL_COMPLETED)
                for th in threads:
                    if isinstance(th.result(), Exception):
                        raise th.result()
                logging.debug(str([t.result() for t in threads]))

            assert len(chunks) - end == n_clusters, "{} vs. {}".format(len(chunks) - end, n_clusters)
            labels.extend(lbls)
            layers.append((end, len(chunks)))
            if callback:
                callback(msg="Cluster one layer: {} -> {}".format(end - start, len(chunks) - end))
            start = end
            end = len(chunks)

        return chunks

