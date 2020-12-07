import torch
import torch.nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
from pdb import set_trace as br
import random

filename = ["selected_DAVAM.txt","selected_GAVAM.txt","selected_GVAM.txt","selected_pretrainVAE.txt"]
      

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()


def preprocess(query):
   query = query.replace('_UNK', '<|endoftext|>')
   query = query.replace('[', '')
   query = query.replace(']', '')
   query = query.replace('<s>', '')
   query = query.replace('</s>', '')
   return query


def GVAM_queries():
    scores_list = []
    sent_score = open("sentences_scores.txt","w")
    for file in filename:
        queries = []
        init_queries = []
        print("file:",file)
        sent_score.write("#####################################")
        sent_score.write("model: %s"%file)
        sent_score.write("#####################################")
        with open(file,'r') as f:
            for line in f.readlines():
                init_queries.append(line)
                line = preprocess(line)
                queries.append(line)
        scores = evaluate(queries, verbose=False)
        print(scores)
        for s, l in zip(scores, init_queries):
           print('score: %.4f, query: %s ' % (s, l))
           sent_score.write('score: %.4f, query: %s ' % (s, l))
        #scores_list.append(new_scores)
        #print('GVAM Len %d, Avg score (ppl.): %.5f, Std: %.5f' % (query_len, np.mean(new_scores[0:100]), np.std(new_scores[0:100])))
    sent_score.close()
    return scores_list


def evaluate(queries, verbose=True):
  perplexities = []
  with torch.no_grad():
    for idx, query in enumerate(queries):
      #print("query:",query)
      new_query = tokenizer.bos_token + ' ' + query
      indexed_tokens = tokenizer.encode(new_query)
      tokens_tensor = torch.tensor([indexed_tokens])

      outputs = model(tokens_tensor)
      predictions = outputs[0]

      softmax = torch.nn.Softmax(dim=-1)
      prob = softmax(predictions[0])

      assert prob.dim() == 2
      assert prob.size()[0] == len(indexed_tokens)

      perplexity = 0
      for i,t in enumerate(indexed_tokens[1:]):
        perplexity += -1 * np.log(prob[i][t].item())
      if len(indexed_tokens)-1>0:
         perplexity = perplexity / (len(indexed_tokens)-1)
         perplexities.append(perplexity)
      else:
         perplexities.append(0)
      #print("perplexity:",perplexity)
      if verbose:
        print('the %d-th query is: %s' % (idx, new_query))
        print ('the tokenized query length: ', len(indexed_tokens))
        print ('the perplexity is: ', perplexity)

  return perplexities


if __name__ == '__main__':
    GVAM_queries()


