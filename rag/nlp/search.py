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
from dataclasses import dataclass

from rag.settings import TAG_FLD, PAGERANK_FLD
from rag.utils import rmSpace
from rag.nlp import rag_tokenizer, query
import numpy as np
from rag.utils.doc_store_conn import DocStoreConnection, MatchDenseExpr, FusionExpr, OrderByExpr


def index_name(uid): return f"ragflow_{uid}"


class Dealer:
    def __init__(self, dataStore: DocStoreConnection):
        self.qryr = query.FulltextQueryer()
        self.dataStore = dataStore

    @dataclass
    class SearchResult:
        total: int
        ids: list[str]
        query_vector: list[float] | None = None
        field: dict | None = None
        highlight: dict | None = None
        aggregation: list | dict | None = None
        keywords: list[str] | None = None
        group_docs: list[list] | None = None

    def get_vector(self, txt, emb_mdl, topk=10, similarity=0.1):
        """
        获取文本的向量表示。

        参数:
        txt (str): 输入文本。
        emb_mdl: 嵌入模型。
        topk (int): 返回的最相似结果的数量。
        similarity (float): 相似度阈值。

        返回:
        MatchDenseExpr: 匹配密集表达式。
        """
        qv, _ = emb_mdl.encode_queries(txt)
        shape = np.array(qv).shape
        if len(shape) > 1:
            raise Exception(
                f"Dealer.get_vector returned array's shape {shape} doesn't match expectation(exact one dimension).")
        embedding_data = [float(v) for v in qv]
        vector_column_name = f"q_{len(embedding_data)}_vec"
        return MatchDenseExpr(vector_column_name, embedding_data, 'float', 'cosine', topk, {"similarity": similarity})

    def get_filters(self, req):
        condition = dict()
        for key, field in {"kb_ids": "kb_id", "doc_ids": "doc_id"}.items():
            if key in req and req[key] is not None:
                condition[field] = req[key]
        # TODO(yzc): `available_int` is nullable however infinity doesn't support nullable columns.
        for key in ["knowledge_graph_kwd", "available_int"]:
            if key in req and req[key] is not None:
                condition[key] = req[key]
        return condition

    def search(self, req, idx_names: str | list[str],
               kb_ids: list[str],
               emb_mdl=None,
               highlight=False,
               rank_feature: dict | None = None
               ):
        """
        执行搜索操作。
    
        参数:
        req (dict): 请求参数。
        idx_names (str | list[str]): 索引名称。
        kb_ids (list[str]): 知识库ID列表。
        emb_mdl: 嵌入模型。
        highlight (bool): 是否高亮显示。
        rank_feature (dict | None): 排序特征。
    
        返回:
        SearchResult: 搜索结果。
        """
        # 获取过滤条件
        filters = self.get_filters(req)
        # 初始化排序表达式
        orderBy = OrderByExpr()
    
        # 解析分页参数
        pg = int(req.get("page", 1)) - 1
        topk = int(req.get("topk", 1024))
        ps = int(req.get("size", topk))
        offset, limit = pg * ps, ps
    
        # ES需要返回哪些字段
        src = req.get("fields",
                      ["docnm_kwd", "content_ltks", "kb_id", "img_id", "title_tks", "important_kwd", "position_int",
                       "doc_id", "page_num_int", "top_int", "create_timestamp_flt", "knowledge_graph_kwd",
                       "question_kwd", "question_tks",
                       "available_int", "content_with_weight", PAGERANK_FLD, TAG_FLD])
        # 初始化关键词集合
        kwds = set([])
    
        # 获取查询问题
        qst = req.get("question", "")
        # 初始化问题向量
        q_vec = []
        if not qst:
            # 如果是列出某个文档的所有分块的请求，top_int好像没有值，要设置成分块的位置索引
            if req.get("sort"):
                orderBy.asc("page_num_int")
                orderBy.asc("top_int")
                orderBy.desc("create_timestamp_flt")
            res = self.dataStore.search(src, [], filters, [], orderBy, offset, limit, idx_names, kb_ids)
            total = self.dataStore.getTotal(res)
            logging.debug("Dealer.search TOTAL: {}".format(total))
        else:
            # 如果有查询问题，进行全文搜索
            highlightFields = ["content_ltks", "title_tks"] if highlight else []
            # matchText：全文检索内容   keywords：关键词列表 见 0005.md
            matchText, keywords = self.qryr.question(qst, min_match=0.3)
            if emb_mdl is None:
                matchExprs = [matchText]
                res = self.dataStore.search(src, highlightFields, filters, matchExprs, orderBy, offset, limit,
                                            idx_names, kb_ids, rank_feature=rank_feature)
                total = self.dataStore.getTotal(res)
                logging.debug("Dealer.search TOTAL: {}".format(total))
            else:
                # 使用嵌入模型进行向量搜索 - 获取问题的向量表示
                matchDense = self.get_vector(qst, emb_mdl, topk, req.get("similarity", 0.1))
                # 从向量表示中提取嵌入数据
                q_vec = matchDense.embedding_data
                # 将问题向量的标识添加到源列表中
                src.append(f"q_{len(q_vec)}_vec")
                
                # 创建融合检索表达式，使用matchText占0.05，向量匹配占0.95
                fusionExpr = FusionExpr("weighted_sum", topk, {"weights": "0.05, 0.95"})
                # 构建匹配表达式列表，包括文本匹配、密集向量匹配和融合表达式
                matchExprs = [matchText, matchDense, fusionExpr]
                
                # 在数据存储中执行搜索,res 的内容见0005.md
                res = self.dataStore.search(src, highlightFields, filters, matchExprs, orderBy, offset, limit,
                                            idx_names, kb_ids, rank_feature=rank_feature)
                # 获取搜索结果的总数
                total = self.dataStore.getTotal(res)
                # 记录搜索结果总数的日志信息
                logging.debug("Dealer.search TOTAL: {}".format(total))
    
                # 如果结果为空，尝试降低匹配阈值再次搜索
                if total == 0:
                    matchText, _ = self.qryr.question(qst, min_match=0.1)
                    filters.pop("doc_ids", None)
                    matchDense.extra_options["similarity"] = 0.17
                    res = self.dataStore.search(src, highlightFields, filters, [matchText, matchDense, fusionExpr],
                                                orderBy, offset, limit, idx_names, kb_ids, rank_feature=rank_feature)
                    total = self.dataStore.getTotal(res)
                    logging.debug("Dealer.search 2 TOTAL: {}".format(total))
    
            # 收集关键词
            for k in keywords:
                kwds.add(k)
                for kk in rag_tokenizer.fine_grained_tokenize(k).split():
                    if len(kk) < 2:
                        continue
                    if kk in kwds:
                        continue
                    kwds.add(kk)
    
        logging.debug(f"TOTAL: {total}")
        ids = self.dataStore.getChunkIds(res)
        keywords = list(kwds)
        highlight = self.dataStore.getHighlight(res, keywords, "content_with_weight")
        aggs = self.dataStore.getAggregation(res, "docnm_kwd")
        return self.SearchResult(
            total=total,
            ids=ids,
            query_vector=q_vec,
            aggregation=aggs,
            highlight=highlight,
            field=self.dataStore.getFields(res, src),
            keywords=keywords
        )
    @staticmethod
    def trans2floats(txt):
        return [float(t) for t in txt.split("\t")]

    def insert_citations(self, answer, chunks, chunk_v,
                         embd_mdl, tkweight=0.1, vtweight=0.9):
        """
        插入引用。

        参数:
        answer (str): 回答文本。
        chunks (list): 文本块列表。
        chunk_v (list): 文本块向量列表。
        embd_mdl: 嵌入模型。
        tkweight (float): 词权重。
        vtweight (float): 向量权重。

        返回:
        tuple: 修改后的回答文本和引用集合。
        """
        assert len(chunks) == len(chunk_v)
        if not chunks:
            return answer, set([])
        pieces = re.split(r"(```)", answer)
        if len(pieces) >= 3:
            i = 0
            pieces_ = []
            while i < len(pieces):
                if pieces[i] == "```":
                    st = i
                    i += 1
                    while i < len(pieces) and pieces[i] != "```":
                        i += 1
                    if i < len(pieces):
                        i += 1
                    pieces_.append("".join(pieces[st: i]) + "\n")
                else:
                    pieces_.extend(
                        re.split(
                            r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])",
                            pieces[i]))
                    i += 1
            pieces = pieces_
        else:
            pieces = re.split(r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])", answer)
        for i in range(1, len(pieces)):
            if re.match(r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])", pieces[i]):
                pieces[i - 1] += pieces[i][0]
                pieces[i] = pieces[i][1:]
        idx = []
        pieces_ = []
        for i, t in enumerate(pieces):
            if len(t) < 5:
                continue
            idx.append(i)
            pieces_.append(t)
        logging.debug("{} => {}".format(answer, pieces_))
        if not pieces_:
            return answer, set([])

        ans_v, _ = embd_mdl.encode(pieces_)
        assert len(ans_v[0]) == len(chunk_v[0]), "The dimension of query and chunk do not match: {} vs. {}".format(
            len(ans_v[0]), len(chunk_v[0]))

        chunks_tks = [rag_tokenizer.tokenize(self.qryr.rmWWW(ck)).split()
                      for ck in chunks]
        cites = {}
        thr = 0.63
        while thr > 0.3 and len(cites.keys()) == 0 and pieces_ and chunks_tks:
            for i, a in enumerate(pieces_):
                sim, tksim, vtsim = self.qryr.hybrid_similarity(ans_v[i],
                                                                chunk_v,
                                                                rag_tokenizer.tokenize(
                                                                    self.qryr.rmWWW(pieces_[i])).split(),
                                                                chunks_tks,
                                                                tkweight, vtweight)
                mx = np.max(sim) * 0.99
                logging.debug("{} SIM: {}".format(pieces_[i], mx))
                if mx < thr:
                    continue
                cites[idx[i]] = list(
                    set([str(ii) for ii in range(len(chunk_v)) if sim[ii] > mx]))[:4]
            thr *= 0.8

        res = ""
        seted = set([])
        for i, p in enumerate(pieces):
            res += p
            if i not in idx:
                continue
            if i not in cites:
                continue
            for c in cites[i]:
                assert int(c) < len(chunk_v)
            for c in cites[i]:
                if c in seted:
                    continue
                res += f" ##{c}$$"
                seted.add(c)

        return res, seted

    def _rank_feature_scores(self, query_rfea, search_res):
        """
        计算排名特征分数。

        参数:
        query_rfea (dict): 查询排名特征。
        search_res (SearchResult): 搜索结果。

        返回:
        np.array: 排名特征分数数组。
        """
        ## For rank feature(tag_fea) scores.
        rank_fea = []
        pageranks = []
        for chunk_id in search_res.ids:
            pageranks.append(search_res.field[chunk_id].get(PAGERANK_FLD, 0))
        pageranks = np.array(pageranks, dtype=float)

        if not query_rfea:
            return np.array([0 for _ in range(len(search_res.ids))]) + pageranks

        q_denor = np.sqrt(np.sum([s*s for t,s in query_rfea.items() if t != PAGERANK_FLD]))
        for i in search_res.ids:
            nor, denor = 0, 0
            for t, sc in eval(search_res.field[i].get(TAG_FLD, "{}")).items():
                if t in query_rfea:
                    nor += query_rfea[t] * sc
                denor += sc * sc
            if denor == 0:
                rank_fea.append(0)
            else:
                rank_fea.append(nor/np.sqrt(denor)/q_denor)
        return np.array(rank_fea)*10. + pageranks

    def rerank(self, sres, query, tkweight=0.3,
               vtweight=0.7, cfield="content_ltks",
               rank_feature: dict | None = None
               ):
        """
        重新排序搜索结果。

        参数:
        sres (SearchResult): 搜索结果。
        query (str): 查询文本。
        tkweight (float): 词权重。
        vtweight (float): 向量权重。
        cfield (str): 内容字段。
        rank_feature (dict | None): 排名特征。

        返回:
        tuple: 相似度数组、词相似度数组、向量相似度数组。
        """
        _, keywords = self.qryr.question(query)
        vector_size = len(sres.query_vector)
        vector_column = f"q_{vector_size}_vec"
        zero_vector = [0.0] * vector_size
        ins_embd = []
        for chunk_id in sres.ids:
            vector = sres.field[chunk_id].get(vector_column, zero_vector)
            if isinstance(vector, str):
                vector = [float(v) for v in vector.split("\t")]
            ins_embd.append(vector)
        if not ins_embd:
            return [], [], []

        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]
        ins_tw = []
        for i in sres.ids:
            content_ltks = sres.field[i][cfield].split()
            title_tks = [t for t in sres.field[i].get("title_tks", "").split() if t]
            question_tks = [t for t in sres.field[i].get("question_tks", "").split() if t]
            important_kwd = sres.field[i].get("important_kwd", [])
            tks = content_ltks + title_tks * 2 + important_kwd * 5 + question_tks * 6
            ins_tw.append(tks)

        ## For rank feature(tag_fea) scores.
        rank_fea = self._rank_feature_scores(rank_feature, sres)

        sim, tksim, vtsim = self.qryr.hybrid_similarity(sres.query_vector,
                                                        ins_embd,
                                                        keywords,
                                                        ins_tw, tkweight, vtweight)

        return sim + rank_fea, tksim, vtsim

    def rerank_by_model(self, rerank_mdl, sres, query, tkweight=0.3,
                        vtweight=0.7, cfield="content_ltks",
                        rank_feature: dict | None = None):
        """
        使用模型重新排序搜索结果。

        参数:
        rerank_mdl: 重新排序模型。
        sres (SearchResult): 搜索结果。
        query (str): 查询文本。
        tkweight (float): 词权重。
        vtweight (float): 向量权重。
        cfield (str): 内容字段。
        rank_feature (dict | None): 排名特征。

        返回:
        tuple: 相似度数组、词相似度数组、向量相似度数组。
        """
        _, keywords = self.qryr.question(query)

        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]
        ins_tw = []
        for i in sres.ids:
            content_ltks = sres.field[i][cfield].split()
            title_tks = [t for t in sres.field[i].get("title_tks", "").split() if t]
            important_kwd = sres.field[i].get("important_kwd", [])
            tks = content_ltks + title_tks + important_kwd
            ins_tw.append(tks)

        tksim = self.qryr.token_similarity(keywords, ins_tw)
        vtsim, _ = rerank_mdl.similarity(query, [rmSpace(" ".join(tks)) for tks in ins_tw])
        ## For rank feature(tag_fea) scores.
        rank_fea = self._rank_feature_scores(rank_feature, sres)

        return tkweight * (np.array(tksim)+rank_fea) + vtweight * vtsim, tksim, vtsim

    def hybrid_similarity(self, ans_embd, ins_embd, ans, inst):
        """
        计算混合相似度。

        参数:
        ans_embd (list): 答案嵌入向量。
        ins_embd (list): 实例嵌入向量。
        ans (str): 答案文本。
        inst (str): 实例文本。

        返回:
        tuple: 相似度数组、词相似度数组、向量相似度数组。
        """
        return self.qryr.hybrid_similarity(ans_embd,
                                           ins_embd,
                                           rag_tokenizer.tokenize(ans).split(),
                                           rag_tokenizer.tokenize(inst).split())

    def retrieval(self, question, embd_mdl, tenant_ids, kb_ids, page, page_size, similarity_threshold=0.2,
                  vector_similarity_weight=0.3, top=1024, doc_ids=None, aggs=True,
                  rerank_mdl=None, highlight=False,
                  rank_feature: dict | None = {PAGERANK_FLD: 10}):
        """
        执行检索操作。
    
        参数:
        question (str): 查询文本。
        embd_mdl: 嵌入模型。
        tenant_ids (str | list[str]): 租户ID。
        kb_ids (list[str]): 知识库ID列表。
        page (int): 页码。
        page_size (int): 每页大小。
        similarity_threshold (float): 相似度阈值。
        vector_similarity_weight (float): 向量相似度权重。
        top (int): 返回的最相似结果的数量。
        doc_ids (list[str] | None): 文档ID列表。
        aggs (bool): 是否聚合。
        rerank_mdl: 重新排序模型。
        highlight (bool): 是否高亮显示。
        rank_feature (dict | None): 排名特征。
    
        返回:
        dict: 检索结果。
        """
        # 初始化排名信息
        ranks = {"total": 0, "chunks": [], "doc_aggs": {}}
        if not question:
            return ranks
    
        # 定义重新排序的页面限制
        RERANK_PAGE_LIMIT = 3
        # 构建检索请求
        req = {"kb_ids": kb_ids, "doc_ids": doc_ids, "size": max(page_size * RERANK_PAGE_LIMIT, 128),
               "question": question, "vector": True, "topk": top,
               "similarity": similarity_threshold,
               "available_int": 1}
    
        # 调整请求参数以适应分页
        if page > RERANK_PAGE_LIMIT:
            req["page"] = page
            req["size"] = page_size
    
        # 处理租户ID
        if isinstance(tenant_ids, str):
            tenant_ids = tenant_ids.split(",")
    
        # 执行搜索
        sres = self.search(req, [index_name(tid) for tid in tenant_ids],
                           kb_ids, embd_mdl, highlight, rank_feature=rank_feature)
        ranks["total"] = sres.total
    
        # 根据模型重新排序
        if page <= RERANK_PAGE_LIMIT:
            if rerank_mdl and sres.total > 0:
                sim, tsim, vsim = self.rerank_by_model(rerank_mdl,
                                                       sres, question, 1 - vector_similarity_weight,
                                                       vector_similarity_weight,
                                                       rank_feature=rank_feature)
            else:
                sim, tsim, vsim = self.rerank(
                    sres, question, 1 - vector_similarity_weight, vector_similarity_weight,
                    rank_feature=rank_feature)
            idx = np.argsort(sim * -1)[(page - 1) * page_size:page * page_size]
        else:
            sim = tsim = vsim = [1] * len(sres.ids)
            idx = list(range(len(sres.ids)))
    
        # 初始化向量相关变量
        dim = len(sres.query_vector)
        vector_column = f"q_{dim}_vec"
        zero_vector = [0.0] * dim
    
        # 构建检索结果
        for i in idx:
            if sim[i] < similarity_threshold:
                break
            if len(ranks["chunks"]) >= page_size:
                if aggs:
                    continue
                break
            id = sres.ids[i]
            chunk = sres.field[id]
            dnm = chunk["docnm_kwd"]
            did = chunk["doc_id"]
            position_int = chunk.get("position_int", [])
            d = {
                "chunk_id": id,
                "content_ltks": chunk["content_ltks"],
                "content_with_weight": chunk["content_with_weight"],
                "doc_id": chunk["doc_id"],
                "docnm_kwd": dnm,
                "kb_id": chunk["kb_id"],
                "important_kwd": chunk.get("important_kwd", []),
                "image_id": chunk.get("img_id", ""),
                "similarity": sim[i],
                "vector_similarity": vsim[i],
                "term_similarity": tsim[i],
                "vector": chunk.get(vector_column, zero_vector),
                "positions": position_int,
            }
            if highlight and sres.highlight:
                if id in sres.highlight:
                    d["highlight"] = rmSpace(sres.highlight[id])
                else:
                    d["highlight"] = d["content_with_weight"]
            ranks["chunks"].append(d)
            if dnm not in ranks["doc_aggs"]:
                ranks["doc_aggs"][dnm] = {"doc_id": did, "count": 0}
            ranks["doc_aggs"][dnm]["count"] += 1
    
        # 排序并限制结果数量
        ranks["doc_aggs"] = [{"doc_name": k,
                              "doc_id": v["doc_id"],
                              "count": v["count"]} for k,
                                                       v in sorted(ranks["doc_aggs"].items(),
                                                                   key=lambda x: x[1]["count"] * -1)]
        ranks["chunks"] = ranks["chunks"][:page_size]

        return ranks
    
    def sql_retrieval(self, sql, fetch_size=128, format="json"):
        """
        执行SQL检索。

        参数:
        sql (str): SQL查询语句。
        fetch_size (int): 获取的记录数。
        format (str): 返回结果的格式。

        返回:
        dict: SQL检索结果。
        """
        tbl = self.dataStore.sql(sql, fetch_size, format)
        return tbl

    def chunk_list(self, doc_id: str, tenant_id: str,
                   kb_ids: list[str], max_count=1024,
                   offset=0,
                   fields=["docnm_kwd", "content_with_weight", "img_id"]):
        """
        获取文档块列表。

        参数:
        doc_id (str): 文档ID。
        tenant_id (str): 租户ID。
        kb_ids (list[str]): 知识库ID列表。
        max_count (int): 最大记录数。
        offset (int): 偏移量。
        fields (list[str]): 字段列表。

        返回:
        list: 文档块列表。
        """
        condition = {"doc_id": doc_id}
        res = []
        bs = 128
        for p in range(offset, max_count, bs):
            es_res = self.dataStore.search(fields, [], condition, [], OrderByExpr(), p, bs, index_name(tenant_id),
                                           kb_ids)
            dict_chunks = self.dataStore.getFields(es_res, fields)
            if dict_chunks:
                res.extend(dict_chunks.values())
            if len(dict_chunks.values()) < bs:
                break
        return res

    def all_tags(self, tenant_id: str, kb_ids: list[str], S=1000):
        """
        获取所有标签。

        参数:
        tenant_id (str): 租户ID。
        kb_ids (list[str]): 知识库ID列表。
        S (int): 参数S。

        返回:
        dict: 标签聚合结果。
        """
        res = self.dataStore.search([], [], {}, [], OrderByExpr(), 0, 0, index_name(tenant_id), kb_ids, ["tag_kwd"])
        return self.dataStore.getAggregation(res, "tag_kwd")

    def all_tags_in_portion(self, tenant_id: str, kb_ids: list[str], S=1000):
        """
        获取部分标签。

        参数:
        tenant_id (str): 租户ID。
        kb_ids (list[str]): 知识库ID列表。
        S (int): 参数S。

        返回:
        dict: 标签聚合结果。
        """
        res = self.dataStore.search([], [], {}, [], OrderByExpr(), 0, 0, index_name(tenant_id), kb_ids, ["tag_kwd"])
        res = self.dataStore.getAggregation(res, "tag_kwd")
        total = np.sum([c for _, c in res])
        return {t: (c + 1) / (total + S) for t, c in res}

    def tag_content(self, tenant_id: str, kb_ids: list[str], doc, all_tags, topn_tags=3, keywords_topn=30, S=1000):
        """
        标记内容。

        参数:
        tenant_id (str): 租户ID。
        kb_ids (list[str]): 知识库ID列表。
        doc (dict): 文档内容。
        all_tags (dict): 所有标签。
        topn_tags (int): 顶级标签数量。
        keywords_topn (int): 关键词顶级数量。
        S (int): 参数S。

        返回:
        bool: 是否成功标记。
        """
        idx_nm = index_name(tenant_id)
        match_txt = self.qryr.paragraph(doc["title_tks"] + " " + doc["content_ltks"], doc.get("important_kwd", []), keywords_topn)
        res = self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nm, kb_ids, ["tag_kwd"])
        aggs = self.dataStore.getAggregation(res, "tag_kwd")
        if not aggs:
            return False
        cnt = np.sum([c for _, c in aggs])
        tag_fea = sorted([(a, round(0.1*(c + 1) / (cnt + S) / (all_tags.get(a, 0.0001)))) for a, c in aggs],
                         key=lambda x: x[1] * -1)[:topn_tags]
        doc[TAG_FLD] = {a: c for a, c in tag_fea if c > 0}
        return True

    def tag_query(self, question: str, tenant_ids: str | list[str], kb_ids: list[str], all_tags, topn_tags=3, S=1000):
        """
        标记查询。

        参数:
        question (str): 查询文本。
        tenant_ids (str | list[str]): 租户ID。
        kb_ids (list[str]): 知识库ID列表。
        all_tags (dict): 所有标签。
        topn_tags (int): 顶级标签数量。
        S (int): 参数S。

        返回:
        dict: 标签聚合结果。
        """
        if isinstance(tenant_ids, str):
            idx_nms = index_name(tenant_ids)
        else:
            idx_nms = [index_name(tid) for tid in tenant_ids]
        match_txt, _ = self.qryr.question(question, min_match=0.0)
        res = self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nms, kb_ids, ["tag_kwd"])
        aggs = self.dataStore.getAggregation(res, "tag_kwd")
        if not aggs:
            return {}
        cnt = np.sum([c for _, c in aggs])
        tag_fea = sorted([(a, round(0.1*(c + 1) / (cnt + S) / (all_tags.get(a, 0.0001)))) for a, c in aggs],
                         key=lambda x: x[1] * -1)[:topn_tags]
        return {a: c for a, c in tag_fea if c > 0}
