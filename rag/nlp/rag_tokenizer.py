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

import copy
import datrie
import math
import os
import re
import string
import sys
from hanziconv import HanziConv
from huggingface_hub import snapshot_download
from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from api.utils.file_utils import get_project_base_directory


class RagTokenizer:
    def key_(self, line):
        return str(line.lower().encode("utf-8"))[2:-1]

    def rkey_(self, line):
        return str(("DD" + (line[::-1].lower())).encode("utf-8"))[2:-1]

    def loadDict_(self, fnm):
        print("[HUQIE]:Build trie", fnm, file=sys.stderr)
        try:
            of = open(fnm, "r", encoding='utf-8')
            while True:
                line = of.readline()
                if not line:
                    break
                line = re.sub(r"[\r\n]+", "", line)
                line = re.split(r"[ \t]", line)
                k = self.key_(line[0])
                F = int(math.log(float(line[1]) / self.DENOMINATOR) + .5)
                if k not in self.trie_ or self.trie_[k][0] < F:
                    self.trie_[self.key_(line[0])] = (F, line[2])
                self.trie_[self.rkey_(line[0])] = 1
            self.trie_.save(fnm + ".trie")
            of.close()
        except Exception as e:
            print("[HUQIE]:Faild to build trie, ", fnm, e, file=sys.stderr)

    def __init__(self, debug=False):
        self.DEBUG = debug
        self.DENOMINATOR = 1000000
        self.trie_ = datrie.Trie(string.printable)
        self.DIR_ = os.path.join(get_project_base_directory(), "rag/res", "huqie")

        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        self.SPLIT_CHAR = r"([ ,\.<>/?;:'\[\]\\`!@#$%^&*\(\)\{\}\|_+=《》，。？、；‘’：“”【】~！￥%……（）——-]+|[a-z\.-]+|[0-9,\.-]+)"
        try:
            self.trie_ = datrie.Trie.load(self.DIR_ + ".txt.trie")
            return
        except Exception as e:
            print("[HUQIE]:Build default trie", file=sys.stderr)
            self.trie_ = datrie.Trie(string.printable)

        self.loadDict_(self.DIR_ + ".txt")

    def loadUserDict(self, fnm):
        try:
            self.trie_ = datrie.Trie.load(fnm + ".trie")
            return
        except Exception as e:
            self.trie_ = datrie.Trie(string.printable)
        self.loadDict_(fnm)

    def addUserDict(self, fnm):
        self.loadDict_(fnm)

    def _strQ2B(self, ustring):
        """把字符串全角转半角"""
        rstring = ""
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 0x3000:
                inside_code = 0x0020
            else:
                inside_code -= 0xfee0
            if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
                rstring += uchar
            else:
                rstring += chr(inside_code)
        return rstring

    def _tradi2simp(self, line):
        return HanziConv.toSimplified(line)

    def dfs_(self, chars, s, preTks, tkslist):
        MAX_L = 10
        res = s
        # if s > MAX_L or s>= len(chars):
        if s >= len(chars):
            tkslist.append(preTks)
            return res

        # pruning
        S = s + 1
        if s + 2 <= len(chars):
            t1, t2 = "".join(chars[s:s + 1]), "".join(chars[s:s + 2])
            if self.trie_.has_keys_with_prefix(self.key_(t1)) and not self.trie_.has_keys_with_prefix(
                    self.key_(t2)):
                S = s + 2
        if len(preTks) > 2 and len(
                preTks[-1][0]) == 1 and len(preTks[-2][0]) == 1 and len(preTks[-3][0]) == 1:
            t1 = preTks[-1][0] + "".join(chars[s:s + 1])
            if self.trie_.has_keys_with_prefix(self.key_(t1)):
                S = s + 2

        ################
        for e in range(S, len(chars) + 1):
            t = "".join(chars[s:e])
            k = self.key_(t)

            if e > s + 1 and not self.trie_.has_keys_with_prefix(k):
                break

            if k in self.trie_:
                pretks = copy.deepcopy(preTks)
                if k in self.trie_:
                    pretks.append((t, self.trie_[k]))
                else:
                    pretks.append((t, (-12, '')))
                res = max(res, self.dfs_(chars, e, pretks, tkslist))

        if res > s:
            return res

        t = "".join(chars[s:s + 1])
        k = self.key_(t)
        if k in self.trie_:
            preTks.append((t, self.trie_[k]))
        else:
            preTks.append((t, (-12, '')))

        return self.dfs_(chars, s + 1, preTks, tkslist)

    def freq(self, tk):
        k = self.key_(tk)
        if k not in self.trie_:
            return 0
        return int(math.exp(self.trie_[k][0]) * self.DENOMINATOR + 0.5)

    def tag(self, tk):
        k = self.key_(tk)
        if k not in self.trie_:
            return ""
        return self.trie_[k][1]

    def score_(self, tfts):
        B = 30
        F, L, tks = 0, 0, []
        for tk, (freq, tag) in tfts:
            F += freq
            L += 0 if len(tk) < 2 else 1
            tks.append(tk)
        F /= len(tks)
        L /= len(tks)
        if self.DEBUG:
            print("[SC]", tks, len(tks), L, F, B / len(tks) + L + F)
        return tks, B / len(tks) + L + F

    def sortTks_(self, tkslist):
        res = []
        for tfts in tkslist:
            tks, s = self.score_(tfts)
            res.append((tks, s))
        return sorted(res, key=lambda x: x[1], reverse=True)

    def merge_(self, tks):
        patts = [
            (r"[ ]+", " "),
            (r"([0-9\+\.,%\*=-]) ([0-9\+\.,%\*=-])", r"\1\2"),
        ]
        # for p,s in patts: tks = re.sub(p, s, tks)

        # if split chars is part of token
        res = []
        tks = re.sub(r"[ ]+", " ", tks).split(" ")
        s = 0
        while True:
            if s >= len(tks):
                break
            E = s + 1
            for e in range(s + 2, min(len(tks) + 2, s + 6)):
                tk = "".join(tks[s:e])
                if re.search(self.SPLIT_CHAR, tk) and self.freq(tk):
                    E = e
            res.append("".join(tks[s:E]))
            s = E

        return " ".join(res)

    def maxForward_(self, line):
        res = []
        s = 0
        while s < len(line):
            e = s + 1
            t = line[s:e]
            while e < len(line) and self.trie_.has_keys_with_prefix(
                    self.key_(t)):
                e += 1
                t = line[s:e]

            while e - 1 > s and self.key_(t) not in self.trie_:
                e -= 1
                t = line[s:e]

            if self.key_(t) in self.trie_:
                res.append((t, self.trie_[self.key_(t)]))
            else:
                res.append((t, (0, '')))

            s = e

        return self.score_(res)

    def maxBackward_(self, line):
        res = []
        s = len(line) - 1
        while s >= 0:
            e = s + 1
            t = line[s:e]
            while s > 0 and self.trie_.has_keys_with_prefix(self.rkey_(t)):
                s -= 1
                t = line[s:e]

            while s + 1 < e and self.key_(t) not in self.trie_:
                s += 1
                t = line[s:e]

            if self.key_(t) in self.trie_:
                res.append((t, self.trie_[self.key_(t)]))
            else:
                res.append((t, (0, '')))

            s -= 1

        return self.score_(res[::-1])

    def english_normalize_(self, tks):
        return [self.stemmer.stem(self.lemmatizer.lemmatize(t)) if re.match(r"[a-zA-Z_-]+$", t) else t for t in tks]

    def tokenize(self, line):
        """这段代码实现了一个中文与英文混合文本的分词器。它首先对输入的字符串进行了一些预处理，包括转换全角字符为半角字符、转换繁体字为简体字，并将所有字母转为小写。然后根据文本中是否包含中文字符来决定使用不同的分词策略。

对于不含中文字符的纯英文文本，它使用了NLTK库中的`word_tokenize`函数进行分词，接着对每个词进行了词干化（stemming）和词形还原（lemmatization）处理。

对于包含中文字符的文本，代码采用了正向最大匹配（maxForward_）和反向最大匹配（maxBackward_）两种方法进行分词，并选择得分更高的分词结果。如果两种方法得到的结果在某些位置上不同，则会进一步尝试使用深度优先搜索（dfs_）来寻找可能更优的分词方案。最后，通过`sortTks_`函数对候选分词结果进行排序，并选取最优的一个。

处理完所有部分后，还会调用`english_normalize_`函数对结果进行英文规范化处理，最后通过`merge_`函数合并分词结果并返回。

        Args:
            line (_type_): _description_

        Returns:
            _type_: _description_
        """
        line = self._strQ2B(line).lower()
        line = self._tradi2simp(line)
        zh_num = len([1 for c in line if is_chinese(c)])
        if zh_num == 0:
            return " ".join([self.stemmer.stem(self.lemmatizer.lemmatize(t)) for t in word_tokenize(line)])

        arr = re.split(self.SPLIT_CHAR, line)
        res = []
        for L in arr:
            if len(L) < 2 or re.match(
                    r"[a-z\.-]+$", L) or re.match(r"[0-9\.-]+$", L):
                res.append(L)
                continue
            # print(L)

            # use maxforward for the first time
            tks, s = self.maxForward_(L)
            tks1, s1 = self.maxBackward_(L)
            if self.DEBUG:
                print("[FW]", tks, s)
                print("[BW]", tks1, s1)

            diff = [0 for _ in range(max(len(tks1), len(tks)))]
            for i in range(min(len(tks1), len(tks))):
                if tks[i] != tks1[i]:
                    diff[i] = 1

            if s1 > s:
                tks = tks1

            i = 0
            while i < len(tks):
                s = i
                while s < len(tks) and diff[s] == 0:
                    s += 1
                if s == len(tks):
                    res.append(" ".join(tks[i:]))
                    break
                if s > i:
                    res.append(" ".join(tks[i:s]))

                e = s
                while e < len(tks) and e - s < 5 and diff[e] == 1:
                    e += 1

                tkslist = []
                self.dfs_("".join(tks[s:e + 1]), 0, [], tkslist)
                res.append(" ".join(self.sortTks_(tkslist)[0][0]))

                i = e + 1

        res = " ".join(self.english_normalize_(res))
        if self.DEBUG:
            print("[TKS]", self.merge_(res))
        return self.merge_(res)

    def fine_grained_tokenize(self, tks):
        """这个方法 `fine_grained_tokenize` 实现了对已经初步分词后的结果进行细粒度分词的功能。具体来说，它首先检查分词结果中中文词汇的比例，如果中文词汇比例小于20%，则直接对每个分词结果进行以斜杠（`/`）为分隔符的再次分割。对于中文词汇比例较高的情况，则会根据单词长度和其他条件采取不同的处理方式。

### 处理流程

1. **中文词汇比例检查**：
   - 计算输入的分词列表 `tks` 中中文词汇的数量占总词汇数的比例。
   - 如果中文词汇比例低于20%，则认为这是一个主要由非中文词汇组成的文本，此时直接对每个分词结果以斜杠为分隔符进行再次分割。

2. **细粒度分词**：
   - 对于每个分词结果 `tk`，如果其长度小于3或者完全由数字、逗号、点、破折号组成，则直接保留原样。
   - 对于长度大于10的较长词汇，直接作为单个单元加入到结果列表 `res` 中，不进行进一步的分割。
   - 对于其他较短的词汇，使用深度优先搜索（DFS）算法尝试找到更好的分词方案，并存储在 `tkslist` 中。
   - 根据 `tkslist` 中的分词结果，选择最合适的分词方案加入到 `res` 中。这里的选择标准包括但不限于分词结果的长度是否与原词汇相同、分词结果是否全部由英文或符号组成等。

3. **英文规范化**：
   - 在完成所有细粒度分词操作后，调用 `english_normalize_` 方法对结果进行英文规范化处理，这可能包括大小写转换、去除标点等操作。

4. **返回结果**：
   - 最终，将处理后的分词结果以空格连接成一个字符串返回。

### 注意事项

- 这个方法假设存在一些辅助函数，例如 `is_chinese` 用于判断字符是否为中文字符，`dfs_` 用于执行深度优先搜索以尝试不同的分词方案，`sortTks_` 用于对分词结果进行排序以选择最优方案，以及 `english_normalize_` 用于英文规范化处理。
- 此方法特别适用于处理包含大量复合词或特殊格式的文本，比如技术文档、专业术语等，这些文本中可能存在较长的单词或特定的分隔符（如斜杠），需要更精细的分词处理才能准确提取出有意义的信息。

这个细粒度分词方法结合了规则和算法，旨在提高分词的准确性，尤其是在处理复杂文本时。

        Args:
            tks (_type_): _description_

        Returns:
            _type_: _description_
        """
        tks = tks.split(" ")
        zh_num = len([1 for c in tks if c and is_chinese(c[0])])
        if zh_num < len(tks) * 0.2:
            res = []
            for tk in tks:
                res.extend(tk.split("/"))
            return " ".join(res)

        res = []
        for tk in tks:
            if len(tk) < 3 or re.match(r"[0-9,\.-]+$", tk):
                res.append(tk)
                continue
            tkslist = []
            if len(tk) > 10:
                tkslist.append(tk)
            else:
                self.dfs_(tk, 0, [], tkslist)
            if len(tkslist) < 2:
                res.append(tk)
                continue
            stk = self.sortTks_(tkslist)[1][0]
            if len(stk) == len(tk):
                stk = tk
            else:
                if re.match(r"[a-z\.-]+$", tk):
                    for t in stk:
                        if len(t) < 3:
                            stk = tk
                            break
                    else:
                        stk = " ".join(stk)
                else:
                    stk = " ".join(stk)

            res.append(stk)

        return " ".join(self.english_normalize_(res))


def is_chinese(s):
    if s >= u'\u4e00' and s <= u'\u9fa5':
        return True
    else:
        return False


def is_number(s):
    if s >= u'\u0030' and s <= u'\u0039':
        return True
    else:
        return False


def is_alphabet(s):
    if (s >= u'\u0041' and s <= u'\u005a') or (
            s >= u'\u0061' and s <= u'\u007a'):
        return True
    else:
        return False


def naiveQie(txt):
    tks = []
    for t in txt.split(" "):
        if tks and re.match(r".*[a-zA-Z]$", tks[-1]
                            ) and re.match(r".*[a-zA-Z]$", t):
            tks.append(" ")
        tks.append(t)
    return tks


tokenizer = RagTokenizer()
tokenize = tokenizer.tokenize
fine_grained_tokenize = tokenizer.fine_grained_tokenize
tag = tokenizer.tag
freq = tokenizer.freq
loadUserDict = tokenizer.loadUserDict
addUserDict = tokenizer.addUserDict
tradi2simp = tokenizer._tradi2simp
strQ2B = tokenizer._strQ2B

if __name__ == '__main__':
    tknzr = RagTokenizer(debug=True)
    # huqie.addUserDict("/tmp/tmp.new.tks.dict")
    tks = tknzr.tokenize(
        "哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈")
    print(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize(
        "公开征求意见稿提出，境外投资者可使用自有人民币或外汇投资。使用外汇投资的，可通过债券持有人在香港人民币业务清算行及香港地区经批准可进入境内银行间外汇市场进行交易的境外人民币业务参加行（以下统称香港结算行）办理外汇资金兑换。香港结算行由此所产生的头寸可到境内银行间外汇市场平盘。使用外汇投资的，在其投资的债券到期或卖出后，原则上应兑换回外汇。")
    print(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize(
        "多校划片就是一个小区对应多个小学初中，让买了学区房的家庭也不确定到底能上哪个学校。目的是通过这种方式为学区房降温，把就近入学落到实处。南京市长江大桥")
    print(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize(
        "实际上当时他们已经将业务中心偏移到安全部门和针对政府企业的部门 Scripts are compiled and cached aaaaaaaaa")
    print(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize("虽然我不怎么玩")
    print(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize("蓝月亮如何在外资夹击中生存,那是全宇宙最有意思的")
    print(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize(
        "涡轮增压发动机num最大功率,不像别的共享买车锁电子化的手段,我们接过来是否有意义,黄黄爱美食,不过，今天阿奇要讲到的这家农贸市场，说实话，还真蛮有特色的！不仅环境好，还打出了")
    print(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize("这周日你去吗？这周日你有空吗？")
    print(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize("Unity3D开发经验 测试开发工程师 c++双11双11 985 211 ")
    print(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize(
        "数据分析项目经理|数据分析挖掘|数据分析方向|商品数据分析|搜索数据分析 sql python hive tableau Cocos2d-")
    print(tknzr.fine_grained_tokenize(tks))
    if len(sys.argv) < 2:
        sys.exit()
    tknzr.DEBUG = False
    tknzr.loadUserDict(sys.argv[1])
    of = open(sys.argv[2], "r")
    while True:
        line = of.readline()
        if not line:
            break
        print(tknzr.tokenize(line))
    of.close()
