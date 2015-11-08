from numpy import zeros, transpose, asarray, sum,  diag, dot, arccos
from numpy.linalg import norm
import numpy
from scipy.linalg import svd, inv
import matplotlib.pyplot as plt
from pattern.web import Wikipedia
import re, random, pylab
from math import * 
from operator import itemgetter
from pattern.web import URL, Document, plaintext

# stopwords, retreived from http://www.lextek.com/manuals/onix/stopwords1.html

stopwords = ['a', 'about', 'above', 'across', 'after', 'again', 'against', 
'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 
'among', 'an', 'and', 'another', 'any', 'anybody', 'anyone', 'anything', 
'anywhere', 'are', 'area', 'areas', 'around', 'as', 'ask', 'asked', 
'asking', 'asks', 'at', 'away', 'b', 'back', 'backed', 'backing', 'backs', 'be', 
'became', 'because', 'become', 'becomes', 'been', 'before', 'began', 'behind', 
'being', 'beings', 'best', 'better', 'between', 'big', 'both', 'but', 'by', 'c', 
'came', 'can', 'cannot', 'case', 'cases', 'certain', 'certainly', 'clear', 'clearly', 
'come', 'could', 'd', 'did', 'differ', 'different', 'differently', 'do', 'does', 'done', 
'down', 'down', 'downed', 'downing', 'downs', 'during', 'e', 'each', 'early', 'either', 
'end', 'ended', 'ending', 'ends', 'enough', 'even', 'evenly', 'ever', 'every', 'everybody', 
'everyone', 'everything', 'everywhere', 'f', 'face', 'faces', 'fact', 'facts', 'far', 
'felt', 'few', 'find', 'finds', 'first', 'for', 'four', 'from', 'full', 'fully', 
'further', 'furthered', 'furthering', 'furthers', 'g', 'gave', 'general', 'generally', 
'get', 'gets', 'give', 'given', 'gives', 'go', 'going', 'good', 'goods', 'got', 'great', 
'greater', 'greatest', 'group', 'grouped', 'grouping', 'groups', 'h', 'had', 'has', 'have', 
'having', 'he', 'her', 'here', 'herself', 'high', 'high', 'high', 'higher', 'highest', 
'him', 'himself', 'his', 'how', 'however', 'i', 'if', 'important', 'in', 'interest', 
'interested', 'interesting', 'interests', 'into', 'is', 'it', 'its', 'itself', 'j', 
'just', 'k', 'keep', 'keeps', 'kind', 'knew', 'know', 'known', 'knows', 'l', 'large', 'largely', 
'last', 'later', 'latest', 'least', 'less', 'let', 'lets', 'like', 'likely', 'long', 'longer',
'longest', 'm', 'made', 'make', 'making', 'man', 'many', 'may', 'me', 'member', 'members', 
'men', 'might', 'more', 'most', 'mostly', 'mr', 'mrs', 'much', 'must', 'my', 'myself', 
'n', 'necessary', 'need', 'needed', 'needing', 'needs', 'never', 'new', 'new', 'newer',
'newest', 'next', 'no', 'nobody', 'non', 'noone', 'not', 'nothing', 'now', 'nowhere', 
'number', 'numbers', 'o', 'of', 'off', 'often', 'old', 'older', 'oldest', 'on',
'once', 'one', 'only', 'open', 'opened', 'opening', 'opens', 'or', 'order', 
'ordered', 'ordering', 'orders', 'other', 'others', 'our', 'out', 'over', 'p', 
'part', 'parted', 'parting', 'parts', 'per', 'perhaps', 'place', 'places', 'point',
'pointed', 'pointing', 'points', 'possible', 'present', 'presented', 'presenting', 
'presents', 'problem', 'problems', 'put', 'puts', 'q', 'quite', 'r', 'rather', 
'really', 'right', 'right', 'room', 'rooms', 's', 'said', 'same', 'saw', 'say', 
'says', 'second', 'seconds', 'see', 'seem', 'seemed', 'seeming', 'seems', 
'sees', 'several', 'shall', 'she', 'should', 'show', 'showed', 'showing', 
'shows', 'side', 'sides', 'since', 'small', 'smaller', 'smallest', 'so', 
'some', 'somebody', 'someone', 'something', 'somewhere', 'state', 'states', 
'still', 'still', 'such', 'sure', 't', 'take', 'taken', 'than', 'that', 'the', 
'their', 'them', 'then', 'there', 'therefore', 'these', 'they', 'thing', 'things', 
'think', 'thinks', 'this', 'those', 'though', 'thought', 'thoughts', 'three', 
'through', 'thus', 'to', 'today', 'together', 'too', 'took', 'toward', 'turn', 
'turned', 'turning', 'turns', 'two', 'u', 'under', 'until', 'up', 'upon', 
'us', 'use', 'used', 'uses', 'v', 'very', 'w', 'want', 'wanted', 'wanting', 
'wants', 'was', 'way', 'ways', 'we', 'well', 'wells', 'went', 'were', 
'what', 'when', 'where', 'whether', 'which', 'while', 'who', 'whole', 'whose', 
'why', 'will', 'with', 'within', 'without', 'work', 'worked', 'working', 'works', 
'would', 'x', 'y', 'year', 'years', 'yet', 'you', 'young', 'younger', 'youngest', 'your', 
'yours', 'z']


ignore_characters = ''',:'!'''

def compare(query1, query2): # core comparison function. 
    lsa = LSA(stopwords, ignore_characters)
    queries = [lsa.search_wiki(query1), lsa.search_wiki(query2)]
    for q in queries:
        lsa.parse(q)
    lsa.build()
    lsa.calc()
    Vt = lsa.Vt
    S = diag(lsa.S)
    vectors =[(dot(S,Vt[:,0]),dot(S,Vt[:,i])) for i in range(len(Vt))]
    angles = [arccos(dot(a,b)/(norm(a,2)*norm(b,2))) for a,b in vectors[1:]]
    return str(abs(1 - float(angles[0])/float(pi/2)))

def graph(query1, query2):
    lsa = LSA(stopwords, ignore_characters)
    titles = [lsa.search_wiki(query1), lsa.search_wiki(query2)]
    for t in titles:
        lsa.parse(t)
    lsa.build()
    lsa.calc()
    lsa.plotSVD()

## core summarization function.
def summarize(query=None, k=4,url=None):
    j = []
    if url:
        b = URL(url)
        a = Document(b.download(cached=True))
        for b in a.get_elements_by_tagname("p"):
            j.append(plaintext(b.content).encode("utf-8"))
        j = [word for sentence in j for word in sentence.split() if re.match("^[a-zA-Z_-]*$", word) or '.' in word or "'" in word or '"' in word]
        j = ' '.join(j)
        lsa1 = LSA(stopwords, ignore_characters)
        sentences = j.split('.')
        sentences = [sentence for sentence in sentences if len(sentence)>1 and sentence != '']
        for sentence in sentences:
            lsa1.parse(sentence)
    else:
        lsa1 = LSA(stopwords, ignore_characters)
        sentences = query.split('.')
        for sentence in sentences:
            lsa1.parse(sentence)
    lsa1.build()
    lsa1.calc()
    summary =[(sentences[i], norm(dot(diag(lsa1.S),lsa1.Vt[:,b]),2)) for i in range(len(sentences)) for b in range(len(lsa1.Vt))]
    sorted(summary, key=itemgetter(1))
    summary = dict((v[0],v) for v in sorted(summary, key=lambda summary: summary[1])).values()
    return '.'.join([a for a, b in summary][len(summary)-(k):])

## evaluate the summarization. How well does the given summary summarize the query?  
def summarize_evaluation(query=None, url=None, summary=None):
    j=[]
    if url:
        b = URL(url)
        a = Document(b.download(cached=True))
        for b in a.get_elements_by_tagname("p"):
            j.append(plaintext(b.content).encode("utf-8"))
        j = [word for sentence in j for word in sentence.split() if re.match("^[a-zA-Z_-]*$", word) or '.' in word or "'" in word or '"' in word]
        j = ' '.join(j)
        lsa = LSA(stopwords, ignore_characters)
        sentences = j.split('.')
        sentences = [sentence for sentence in sentences if len(sentence)>1 and sentence != '']
        for sentence in sentences:
            lsa.parse(sentence)
    else:
        lsa = LSA(stopwords, ignore_characters)
        for sentence in query:
            lsa.parse(sentence)
    lsa.build()
    lsa.calc()
    lsa2 = LSA(stopwords, ignore_characters)
    for sentence in summary:
        lsa2.parse(sentence)
    lsa2.build()
    lsa2.calc()
    vectors =[(dot(lsa.S,lsa.U[0,:]),dot(lsa.S,lsa.U[i,:])) for i in range(len(lsa.U))]
    vectors2 =[(dot(lsa2.S,lsa2.U[0,:]),dot(lsa2.S,lsa2.U[i,:])) for i in range(len(lsa2.U))]
    angles = [arccos(dot(a,b)/(norm(a,2)*norm(b,2))) for a in vectors for b in vectors2]
    return str(abs(1 - float(angles[1])/float(pi/2)))

class LSA(object):
    def __init__(self, stopwords, ignore_characters):
        self.stopwords = stopwords
        self.ignore_characters = ignore_characters
        self.wdict = {}
        self.dcount = 0        
    def parse(self, doc):
        words = doc.split();
        for w in words:
            w = w.lower().translate(None, self.ignore_characters)
            if w in self.stopwords:
                continue
            elif w in self.wdict:
                self.wdict[w].append(self.dcount)
            else:
                self.wdict[w] = [self.dcount]
        self.dcount += 1      
    def build(self): # Create count matrix
        self.keys = [k for k in self.wdict.keys() if len(self.wdict[k]) > 1]
        self.keys.sort()
        self.A = zeros([len(self.keys), self.dcount])
        for i, k in enumerate(self.keys):
            for d in self.wdict[k]:
                self.A[i,d] += 1
    def calc(self): # execute SVD
        self.U, self.S, self.Vt = svd(self.A, full_matrices =False)
    def TFIDF(self): # calculate tfidf score
        WordsPerDoc = sum(self.A, axis=0)        
        DocsPerWord = sum(asarray(self.A > 0, 'i'), axis=1)
        rows, cols = self.A.shape
        for i in range(rows):
            for j in range(cols):
                self.A[i,j] = (self.A[i,j] / WordsPerDoc[j]) * log(float(cols) / DocsPerWord[i])
    def S(self):
        return self.S
    def U(self):
        return -1 * self.U
    def Vt(self):
        return -1 * self.Vt
    def printSVD(self):
        print 'Singular values: '
        print self.S
        print 'U matrix: '
        print -1*self.U[:, 0:3]
        print 'Vt matrix: '
        print -1*self.Vt[0:3, :]
    def search_wiki(self, k): # scrape query's wikipedia article
        article = Wikipedia().search(k)
        contents = [section.content.encode("utf8") for section in article.sections]
        d = []
        for content in contents:
            a = content.split()
            d.append(a)
        content = [j for i in d for j in i if re.match("^[a-zA-Z_-]*$", j) and len(j) > 1] # take only meaningful content
        self.content = ' '.join(content)
        return self.content
    def plotSVD(self, k = 5): # change k to change how many points you want to see on the graph. plots term vectors vs. document vectors.
        y = numpy.random.random(10)
        d = numpy.random.random(10)
        fig = plt.figure()
        graph = fig.add_subplot(111)
        graph.autoscale(True)
        coordinates  = [(s,a) for [s,a] in (-1 * self.U[:,0:3]).tolist()]
        plot_coordinates = []
        for i in range(k):
            index = random.randint(1,len(coordinates))
            plot_coordinates.append(coordinates[index])
        xdata = [s for s,a in plot_coordinates]
        ydata = [a for s,a in plot_coordinates]
        plt.Arrow(0, 0, xdata[0], ydata[0])
        graph.scatter(xdata,ydata, c = y, s = 20)
        graph.scatter(self.Vt[0:2,:].tolist()[0], self.Vt[0:2,:].tolist()[1],  marker='^', c = d, s = 100)   
        plt.show()



if __name__ == "__main__": 
    print "\n\n ---- TESTS OF SUMMARIZATION ---- \n\n"
    string = 'Principal component analysis (PCA), based on information theory concepts, seeks a computational model that best describes a face by extracting the most relevant information contained in that face. The Eigenfaces approach is a PCA method, in which a small set of characteristic pictures are used to describe the variation between face images. The goal is to find the eigenvectors (eigenfaces) of the covariance matrix of the distribution, spanned by training a set of face images. Later, every face image is represented by a linear combination of these eigenvectors. Recognition is performed by projecting a new image onto the subspace spanned by the eigenfaces and then classifying the face by comparing its position in the face space with the positions of known individuals.'
    print "TASK - SUMMARIZE A DESCRIPTION OF THE GETTYSBURG ADDRESS: \n"
    print string
    print "\n"
    print " 1 SENTENCE SUMMARY:\n"
    summary = summarize(query=string, k = 1)
    print summary + "\n"
    print "TASK - SUMMARIZE I HAVE A DREAM SPEECH: (http://www.huffingtonpost.com/2011/01/17/i-have-a-dream-speech-text_n_809993.html) \n"
    url="http://www.huffingtonpost.com/2011/01/17/i-have-a-dream-speech-text_n_809993.html"
    summary1=summarize(url=url, k=4)
    print "4 SENTENCE SUMMARY:\n"
    print summary1 + "\n"
    print "---- TESTS OF SIMILARITY ---- \n\n"
    print "Similarity between Facebook and Mark Zuckerberg:" + compare('Facebook', 'Mark Zuckerberg')
    print "Similarity between Dick Cheney and George Bush: " + compare ('Dick Cheney', 'George W Bush')
    print "Similarity between Nickelodeon and Genghis Khan: " + compare ('Nickelodeon', 'Genghis Khan')
    print "Similarity between Grapes and Cars: " + compare ('grapes', 'cars')
    print "Similarity between Brad Pitt and Angelina Jolie: " + compare ('Brad Pitt', 'Angelina Jolie')
    print "Similarity between Barack Obama and Michelle Obama: " + compare ('Barack Obama', 'Michelle Obama')

    

