import re

final_ans = """
dhwidfoi347rdfh387d3878d478
- [文件名1.pdf](/viewer/document/7fkfghjwe73r347r63476?ext=pdf&prefix=document)
- [文件名2.pdf](/viewer/document/fge7wif467rt73r234472t67er?ext=pdf&prefix=document)
"""

# 提取 final_ans中，符合 [filename](file_url) 模式的子串，然后将filename部分提取到 filenames ,将file_url部分提取到 file_urls 数组中
files = re.findall(r'\[([^]]+)\]\(([^)]+)\)', final_ans)

# file_urls 的模式是 /viewer/document/{doc_id}?ext={filetype}&prefix=document   ，提取出doc_id 到 doc_ids 数组中
for file in files:
  filename = file[0]
  file_url = file[1]
  doc_id = re.findall(r'/viewer/document/(\S+)', file_url.split("?")[0])[0]
  pass
