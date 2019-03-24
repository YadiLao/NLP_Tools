# NLP_Tools

#### 文字转拼音
pypinyin: pip install pypinyin [link](https://github.com/mozillazg/python-pinyin)

```
from pypinyin import pinyin, lazy_pinyin, Style
from pypinyin import load_phrases_dict, load_single_dict

print(pinyin('中心'))
print(pinyin('中心', heteronym=True)) 
print(pinyin('中心', style=Style.FIRST_LETTER))  
print(pinyin('中心', style=Style.TONE2, heteronym=True))
print(lazy_pinyin('中心'))  
```



#### 模糊匹配
python-Levenshtein

```
ratio = Levenshtein.ratio(s1, s2)
dist = Levenshtein.distance(s1, s2)
```

FuzzyWuzzy: [link](https://github.com/seatgeek/fuzzywuzzy)


#### 中文近义词包

Synonyms: [link](https://github.com/huyingxi/Synonyms)



#### 搜狗联想词爬取

```
def findRelatedWordInSougo(keyword):
    keyword = urllib.request.quote(keyword)
    url = "http://w.sugg.sogou.com/sugg/ajaj_json.jsp?key="+keyword+"&type=web&ori=yes&pr=web&abtestid=1&ipn="
    res = urllib.request.urlopen(url)
    keylist = res.read()
    keylist = keylist.decode('GBK')
    rl = r'"(\D.*?)"'
    rl = re.compile(rl)
    word = re.findall(rl, keylist)
    print(word)
```





------------------
#### ref:
1. funNLP: [link](https://github.com/fighting41love/funNLP)
