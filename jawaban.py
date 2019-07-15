import numpy as np
import pandas as pd

buku = pd.read_csv('books.csv')
rating = pd.read_csv('ratings.csv')

def MergeCol(i):
    return str(i['authors'])+' '+str(i['original_title'])+' '+str(i['title'])+' '+str(i['language_code'])

buku['fitur'] = buku.apply(MergeCol,axis=1)
print(buku.head())

#count vectorizer
from sklearn.feature_extraction.text import CountVectorizer
model = CountVectorizer(tokenizer=lambda i:i.split(' '))
matrixFeature = model.fit_transform(buku['fitur'])

feat = model.get_feature_names()
jml_fitur = len(feat)

#cosinus similarity
from sklearn.metrics.pairwise import cosine_similarity
skor = cosine_similarity(matrixFeature)

# buku masing masing
a1 = buku[buku['original_title']=='The Hunger Games']['book_id'].tolist()[0]-1 
a2 = buku[buku['original_title']=='Catching Fire']['book_id'].tolist()[0]-1 
a3 = buku[buku['original_title']=='Mockingjay']['book_id'].tolist()[0]-1 
a4 = buku[buku['original_title']=='The Hobbit or There and Back Again']['book_id'].tolist()[0]-1 
andi = [a1,a2,a3,a4]

b1 = buku[buku['original_title']=='Harry Potter and the Philosopher\'s Stone']['book_id'].tolist()[0]-1 
b2 = buku[buku['original_title']=='Harry Potter and the Chamber of Secrets']['book_id'].tolist()[0]-1 
b3 = buku[buku['original_title']=='Harry Potter and the Prisoner of Azkaban']['book_id'].tolist()[0]-1 
budi = [b1,b2,b3]

c1 = buku[buku['original_title']=='Robots and Empire']['book_id'].tolist()[0]-1 
ciko = [c1]

d1 = buku[buku['original_title']=='Nine Parts of Desire: The Hidden World of Islamic Women']['book_id'].tolist()[0]-1 
d2 = buku[buku['original_title']=='A History of God: The 4,000-Year Quest of Judaism, Christianity, and Islam']['book_id'].tolist()[0]-1 
d3 = buku[buku['original_title']=='No god but God: The Origins, Evolution, and Future of Islam']['book_id'].tolist()[0]-1 
dedi = [d1,d2,d3]

e1 = buku[buku['original_title']=='Doctor Sleep']['book_id'].tolist()[0]-1 
e2 = buku[buku['original_title']=='The Story of Doctor Dolittle']['book_id'].tolist()[0]-1 
e3 = buku[buku['title']=='Bridget Jones\'s Diary (Bridget Jones, #1)']['book_id'].tolist()[0]-1 
ello = [e1,e2,e3]

skorA1 = list(enumerate(skor[a1]))
skorA2 = list(enumerate(skor[a2]))
skorA3 = list(enumerate(skor[a3]))
skorA4 = list(enumerate(skor[a4]))
skorAndi = []
for i in skorA1:
    skorAndi.append((i[0],0.25*(skorA1[i[0]][1]+skorA2[i[0]][1]+skorA3[i[0]][1]+skorA4[i[0]][1])/4))

skorB1 = list(enumerate(skor[b1]))
skorB2 = list(enumerate(skor[b2]))
skorB3 = list(enumerate(skor[b3]))
skorBudi = []
for i in skorB1:
    skorBudi.append((i[0],(skorB1[i[0]][1]+skorB2[i[0]][1]+skorB3[i[0]][1])/3))

skorCiko = list(enumerate(skor[c1]))

skorD1 = list(enumerate(skor[d1]))
skorD2 = list(enumerate(skor[d2]))
skorD3 = list(enumerate(skor[d3]))
skorDedi = []
for i in skorD1:
    skorDedi.append((i[0],(skorD1[i[0]][1]+skorD2[i[0]][1]+skorD3[i[0]][1])/3))

skorE1 = list(enumerate(skor[e1]))
skorE2 = list(enumerate(skor[e2]))
skorE3 = list(enumerate(skor[e3]))
skorEllo = []
for i in skorE1:
    skorEllo.append((i[0],(skorE1[i[0]][1]+skorE2[i[0]][1]+skorE3[i[0]][1])/3))

#sorting
sortAndi = sorted(skorAndi, key = lambda i:i[1], reverse = True)
sortBudi = sorted(skorBudi, key = lambda i:i[1], reverse = True)
sortCiko = sorted(skorCiko, key = lambda i:i[1], reverse = True)
sortDedi = sorted(skorDedi, key = lambda i:i[1], reverse = True)
sortEllo = sorted(skorEllo, key = lambda i:i[1], reverse = True)

#top 5 
recommendAndi = []
for i in sortAndi:
    if i[1]>0:
        recommendAndi.append(i)

recommendBudi = []
for i in sortBudi:
    if i[1]>0:
        recommendBudi.append(i)

recommendCiko = []
for i in sortCiko:
    if i[1]>0:
        recommendCiko.append(i)

recommendDedi = []
for i in sortDedi:
    if i[1]>0:
        recommendDedi.append(i)

recommendEllo = []
for i in sortEllo:
    if i[1]>0:
        recommendEllo.append(i)


print('1. Buku bagus untuk Andi:')
for i in range(0,5):
    if recommendAndi[i][0] not in andi:
        print('-',buku['original_title'].iloc[recommendAndi[i][0]])
    else:
        i+=5
        print('-',buku['original_title'].iloc[recommendAndi[i][0]])

print('\n2. Buku bagus untuk Budi:')
for i in range(0,5):
    if recommendBudi[i][0] not in budi:
        print('-',buku['original_title'].iloc[recommendBudi[i][0]])
    else:
        i+=5
        print('-',buku['original_title'].iloc[recommendBudi[i][0]])

print('\n3. Buku bagus untuk Ciko:')
for i in range(0,5):
    if recommendCiko[i][0] not in ciko:
        print('-',buku['original_title'].iloc[recommendCiko[i][0]])
    else:
        i+=5
        print('-',buku['original_title'].iloc[recommendCiko[i][0]])

print('\n4. Buku bagus untuk Dedi:')
for i in range(0,5):
    if recommendDedi[i][0] not in dedi:
        print('-',buku['original_title'].iloc[recommendDedi[i][0]])
    else:
        i+=5
        print('-',buku['original_title'].iloc[recommendDedi[i][0]])

print('\n5. Buku bagus untuk Ello:')
for i in range(0,5):
    if recommendEllo[i][0] not in ello:
        if str(buku['original_title'].iloc[recommendEllo[i][0]])=='nan':
            print('-',buku['title'].iloc[recommendEllo[i][0]])
        else:
            print('-',buku['original_title'].iloc[recommendEllo[i][0]])  
    else:
        i+=5
        if str(buku['original_title'].iloc[recommendEllo[i][0]])=='nan':
            print('-',buku['title'].iloc[recommendEllo[i][0]])
        else:
            print('-',buku['original_title'].iloc[recommendEllo[i][0]]) 