import fasttext


#fasttext.skipgram('cooking.stackexchange.txt', 'model')
#model = fasttext.train_supervised('/Users/peter/workspace/cooking.stackexchange.txt', wordNgrams=2, epoch=25, lr=0.5)
model = fasttext.train_supervised('/Users/peter/Repos/train.csv', wordNgrams=2, epoch=25, lr=0.5)


# model = fasttext.load_model("model_cooking.bin")

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

#print_results(*model.test('/Users/peter/workspace/cooking.stackexchange.txt')) 
print_results(*model.test('/Users/peter/Repos/train.csv'))
print (model.predict("Which baking dish is best to bake a banana bread ?"))

#model.save_model("model_cooking.bin")
