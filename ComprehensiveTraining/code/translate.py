import http.client
import hashlib
from urllib import parse
import random
from re import findall

class Translation():
    def __init__(self):
        self.appid = '20180927000213015'
        self.secretKey = '0D1BUT_G0OvlwONKhisO'
        self.httpClient = None
        self.myurl = '/api/trans/vip/translate'
        self.m1 = hashlib.md5()
    def translate(self,query):
        if len(findall("[a-zA-Z]",query))>1:
            fromLang = 'eh'
            toLang = 'zh'
        else:
            fromLang = 'zh'
            toLang = 'en'
        salt = random.randint(32768, 65536)
        
        salt = str(salt)
        sign = self.appid + query + salt+self.secretKey

        self.m1.update(sign.encode(encoding='utf-8'))
        sign = self.m1.hexdigest()
        myurl = self.myurl+'?appid='+self.appid+'&q='+parse.quote(query)+'&from='+fromLang+'&to='+toLang+'&salt='+salt+'&sign='+sign
        try:
            httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
            httpClient.request('GET', myurl)
            response = httpClient.getresponse()
            string_info = response.read().decode('utf-8')
            string_info = eval(string_info)


        except Exception as e:
            print(e)
        finally:
            if httpClient:
                httpClient.close()
        result = []
        print(string_info)
        for line in string_info['trans_result']:
           result.append(line["dst"])
        return result

if __name__ == "__main__":
    one = Translation()
    result = one.translate("操你妈！")

    for answer in result:
        print(answer)