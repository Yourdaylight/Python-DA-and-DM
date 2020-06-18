import os
import pyltp
import jieba
from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer,SementicRoleLabeller
from collections import OrderedDict
region=["区","县","号线"]
jieba.add_word("A市")
jieba.add_word("A市经济学院")
jieba.add_word("经济学院")
jieba.add_word("体育学院")
jieba.add_word("B市")
for i in range(1,10):
    word1="A"+str(i)+"区"
    word2="A"+str(i)+"县"
    word3=str(i)+"号线"
    jieba.add_word(word1)
    jieba.add_word(word2)
    jieba.add_word(word3)
class LtpParser():
    def __init__(self):
        LTP_DIR = "E:\\study\\Projects\\data-mining\\ltp\\ltp_data_v3.4.0"
        self.segmentor = Segmentor()
        #self.segmentor.load_with_lexicon(os.path.join(LTP_DIR, "cws.model"), os.path.join(LTP_DIR, "word_dict")) #加载外部词典

        self.postagger = Postagger()
        self.postagger.load_with_lexicon(os.path.join(LTP_DIR, "pos.model"), os.path.join(LTP_DIR, "n_word_dict")) #加载外部词典

        self.parser = Parser()
        self.parser.load(os.path.join(LTP_DIR, "parser.model")) #依存句法分析

        self.recognizer = NamedEntityRecognizer()
        self.recognizer.load(os.path.join(LTP_DIR, "ner.model"))#实体识别

        self.labeller = SementicRoleLabeller()  # 语义角色标注
        self.labeller.load(os.path.join(LTP_DIR, "pisrl_win.model"))

        #加载停用词
        with open(LTP_DIR + '\\stopwords.txt', 'r', encoding='gbk') as fread:
            self.stopwords = set()
            for line in fread:
                self.stopwords.add(line.strip())

    '''把实体和词性给进行对应'''
    def wordspostags(self, name_entity_dist, words, postags):
        pre = ' '.join([item[0] + '/' + item[1] for item in zip(words, postags)])
        post = pre
        for et, infos in name_entity_dist.items():
            if infos:
                for info in infos:
                    post = post.replace(' '.join(info['consist']), info['name'])
        post = [word for word in post.split(' ') if len(word.split('/')) == 2 and word.split('/')[0]]
        words = [tmp.split('/')[0] for tmp in post]
        postags = [tmp.split('/')[1] for tmp in post]

        return words, postags

    '''根据实体识别结果,整理输出实体列表'''
    def entity(self, words, netags, postags):
        '''
        :param words: 词
        :param netags: 实体
        :param postags: 词性
        :return:
        '''
        name_entity_dict = {}
        name_entity_list = []
        place_entity_list = []
        organization_entity_list = []
        ntag_E_Nh = ""
        ntag_E_Ni = ""
        ntag_E_Ns = ""
        index = 0
        for item in zip(words, netags):
            word = item[0]
            ntag = item[1]
            if ntag[0] != "O":
                if ntag[0] == "S":
                    if ntag[-2:] == "Nh":
                        name_entity_list.append(word + '_%s ' % index)
                    elif ntag[-2:] == "Ni":
                        organization_entity_list.append(word + '_%s ' % index)
                    else:
                        place_entity_list.append(word + '_%s ' % index)
                elif ntag[0] == "B":
                    if ntag[-2:] == "Nh":
                        ntag_E_Nh = ntag_E_Nh + word + '_%s ' % index
                    elif ntag[-2:] == "Ni":
                        ntag_E_Ni = ntag_E_Ni + word + '_%s ' % index
                    else:
                        ntag_E_Ns = ntag_E_Ns + word + '_%s ' % index
                elif ntag[0] == "I":
                    if ntag[-2:] == "Nh":
                        ntag_E_Nh = ntag_E_Nh + word + '_%s ' % index
                    elif ntag[-2:] == "Ni":
                        ntag_E_Ni = ntag_E_Ni + word + '_%s ' % index
                    else:
                        ntag_E_Ns = ntag_E_Ns + word + '_%s ' % index
                else:
                    if ntag[-2:] == "Nh":
                        ntag_E_Nh = ntag_E_Nh + word + '_%s ' % index
                        name_entity_list.append(ntag_E_Nh)
                        ntag_E_Nh = ""
                    elif ntag[-2:] == "Ni":
                        ntag_E_Ni = ntag_E_Ni + word + '_%s ' % index
                        organization_entity_list.append(ntag_E_Ni)
                        ntag_E_Ni = ""
                    else:
                        ntag_E_Ns = ntag_E_Ns + word + '_%s ' % index
                        place_entity_list.append(ntag_E_Ns)
                        ntag_E_Ns = ""
            index += 1
        name_entity_dict['nhs'] = self.modify(name_entity_list, words, postags, 'nh')
        name_entity_dict['nis'] = self.modify(organization_entity_list, words, postags, 'ni')
        name_entity_dict['nss'] = self.modify(place_entity_list, words, postags, 'ns')
        return name_entity_dict

    def modify(self, entity_list, words, postags, tag):
        modify = []
        if entity_list:
            for entity in entity_list:
                entity_dict = {}
                subs = entity.split(' ')[:-1]
                start_index = subs[0].split('_')[1]
                end_index = subs[-1].split('_')[1]
                entity_dict['stat_index'] = start_index
                entity_dict['end_index'] = end_index
                if start_index == entity_dict['end_index']:
                    consist = [words[int(start_index)] + '/' + postags[int(start_index)]]
                else:
                    consist = [words[index] + '/' + postags[index] for index in
                               range(int(start_index), int(end_index) + 1)]
                entity_dict['consist'] = consist
                entity_dict['name'] = ''.join(tmp.split('_')[0] for tmp in subs) + '/' + tag
                modify.append(entity_dict)
        return modify

    '''词性和实体'''
    def post_ner(self, words):
        postags = list(self.postagger.postag(words))
        # words_filter =[]
        # postags = []
        # for word, postag in zip(words, self.postagger.postag(words)):
        #     if 'n' in postag:
        #         postags.append(postag)
        #         words_filter.append(word)
        nerags = self.recognizer.recognize(words, postags)
        return postags, nerags

    def parser_process(self, sentence):
        words = list(jieba.cut(sentence))
        post, ner = self.post_ner(words)  # 词性和实体
        name_entity_dist = self.entity(words, ner, post)
        words, postags = self.wordspostags(name_entity_dist, words, post)
        return words, postags

    def analysis(self,sentence):
        words = list(jieba.cut(sentence))
        post, ner = self.post_ner(words)  # 词性和实体
        arcs = self.parser.parse(words, post)
        return words,arcs

    def getWord(self,Type, arcs, words):
        res = None
        for i in range(len(words)):
            if arcs[i].relation == Type:
                res = words[i]
                break
        return res

    def getFirst(self,List):
        for i in List:
            if i is not None:
                return i
        return None

    def getMain(self,sentence):
        words, arcs = self.analysis(sentence)
        hed = self.getWord("HED", arcs, words)

        sbv =self.getWord("SBV", arcs, words)
        vob =self.getWord("VOB", arcs, words)
        fob =self.getWord("FOB", arcs, words)

        adv =self.getWord("ADV", arcs, words)
        pob =self.getWord("POB", arcs, words)

        zhu =self.getFirst([sbv, pob])
        wei = hed
        bin = self.getFirst([vob, fob, pob])

        string = '{}{}{},（副词：{}）'.format(zhu, wei, bin, adv)
        return string.replace('None', '')

    def release_model(self):
        # 释放模型
        self.segmentor.release()
        self.postagger.release()
        self.recognizer.release()
        self.parser.release()
        # self.labeller.release()
