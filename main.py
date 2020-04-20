# encoding:utf-8
# Filename: using_name.py
import cv2 as cv
from aip import AipOcr
import codecs
import jieba
import jieba.posseg as pseg
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
# import gensim
from gensim import corpora, models, similarities
""" 你的 APPID AK SK """
APP_ID = '15869494'
API_KEY = 'PAwPQfw7BLll8CBMPE0WTzuK'
SECRET_KEY = 'WgnyZgoANRGldCPFROrnr8eDyBU1Qz4n'
client = AipOcr(APP_ID, API_KEY, SECRET_KEY)
# 加载题库
def load_tq():
    tq_data = []
    print('加载题库中.....................')
    f = open("tq.txt","r")   #设置文件对象
    data = f.readlines()  #直接将文件中按行读到list里，效果与方法2一样
    f.close()             #关闭文件
    for row in data:
        line = row.strip()
        line_keyword_list = [word for word in jieba.cut(line)]
        tq_data.append([line,line_keyword_list])
        # print("【题库】："+"/".join(line_keyword_list))
    print('加载题库完成！')
    return tq_data
# 写题库
def write_tq(str):
    f = open("tq_1.txt","a", encoding='UTF-8')   #设置文件对象
    f.writelines("\n"+str)  #直接将文件中按行读到list里，效果与方法2一样
    f.close()             #关闭文件
# 调用摄像头或许图片
def video_demo():
    #0是代表摄像头编号，只有一个的话默认为0
    capture=cv.VideoCapture(0) 
    while(True):
        ref,frame=capture.read()
 
        cv.imshow("1",frame)
    #等待30ms显示图像，若过程中按“S”退出
        c= cv.waitKey(30) & 0xff 
        if c==ord('s'):
            cv.imwrite('1.png',frame)
            capture.release()
            cv.destroyAllWindows()
            break
# 读取图片
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()            
# 调用百度API识别文字
def baiduAPI(image):
    # """ 调用通用文字识别, 图片参数为本地图片 """
    # client.basicGeneral(image);

    # """ 如果有可选参数 """
    options = {}
    options["language_type"] = "CHN_ENG"
    options["detect_direction"] = "true"
    options["detect_language"] = "true"
    options["probability"] = "true"

    # """ 带参数调用通用文字识别, 图片参数为本地图片 """
    rs_orc =client.basicGeneral(image, options)
    # print(rs_orc)  #打印百度接口返回值
    return rs_orc
# 调用jieba分词获得关键词列表
def get_jiebastr(keystr):
    strlist = pseg.cut(keystr)
    return strlist
# 将jieba分词列表进行优化
def get_keywordlist(words_result):
    keywordlist = []
    words_result_str = ""
    for word in words_result:
            words_result_str = words_result_str + word['words']
    strlist = get_jiebastr(words_result_str)
    keywordlist.append(words_result_str)
    # 加载停用词表
    print('加载停用词表中..........')
    stopwords = codecs.open("stopwords.txt",'r',encoding='utf8').readlines()
    stopwords = [ w.strip() for w in stopwords ]
    print('加载完成！')
    # 加载停用词性[]
    stop_flag = ['w']
    for word, flag in strlist:
        # # 去重
        if word not in keywordlist:
            keywordlist.append(word)
        # 鉴别停用词和词性
        # if flag not in stop_flag and word not in stopwords and word not in keywordlist:
        #     keywordlist.append(word)
    return keywordlist
# 程序入口
if __name__ == '__main__':
    print('======================程序开始==========================')
    tq_data = load_tq()
    while(True):
        video_demo()
        cv.waitKey()
        image = get_file_content('1.png')
        orc_data = baiduAPI(image)
        words_result = orc_data['words_result']
        # print(words_result)
        words_len = len(words_result)
        keywordlist = get_keywordlist(words_result)
        question = keywordlist[0]
        print("【题  目】："+ question)
        print("【关键词】："+ "/".join(keywordlist))
        print("========================")
        all_doc_list =[]
        for tq_line in tq_data:
            tq = tq_line[0] #题目
            tq_key = tq_line[1] #题目的分词列表
            all_doc_list.append(tq_key)
        dictionary = corpora.Dictionary(all_doc_list)
        dictionary.keys()
        dictionary.token2id
        corpus = [dictionary.doc2bow(doc) for doc in all_doc_list]
        doc_test_vec = dictionary.doc2bow(keywordlist)
        tfidf = models.TfidfModel(corpus)
        index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))
        sim = index[tfidf[doc_test_vec]]
        sim_sorted =sorted(enumerate(sim), key=lambda item: -item[1])
        for i in range(0,5):
            rs_id = sim_sorted[i][0]
            rs_r = sim_sorted[i][1]
            if rs_r>0 :
                print("【结果】"+ tq_data[rs_id][0]+"["+str(rs_r)+"]")
        key = input('回车键进入下一题，按q退出:')
        if key == 'q':
            print('========================程序退出=========================')
            break
        
        else:
            if key in ['a','b','c','d','e','f']:
                answer = "【"+key.upper()+"】"
                write_tq(question+answer)
            continue
        
else:
    print('None')





