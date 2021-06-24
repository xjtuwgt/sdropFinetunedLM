# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import pandas as pd
from pandas import DataFrame
from time import time
from longformer_retrieval.lb_Longformer_hotpotQAUtils import LongformerTokenizer, normalize_question, normalize_text, \
    query_encoder, context_encoder, test_context_merge_longer
import swifter
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Hotpot_Retrieval_Test_Data_PreProcess(data: DataFrame, tokenizer: LongformerTokenizer):
    def test_normalize_row(row):
        question, context = row['question'], row['context']
        norm_question = normalize_question(question=question)
        ################################################################################################################
        norm_context = []
        for ctx_idx, ctx in enumerate(context):
            ctx_title, ctx_sentences = ctx
            norm_ctx_sentences = [normalize_text(sent) for sent in ctx_sentences]
            norm_context.append([ctx_title.lower(), norm_ctx_sentences])
        ################################################################################################################
        return norm_question, norm_context
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    start_time = time()
    normalized_names = ['norm_question', 'norm_ctx']
    data[normalized_names] = data.swifter.apply(lambda row: pd.Series(test_normalize_row(row)), axis=1)
    print('Step 1: Normalizing data takes {:.4f} seconds'.format(time() - start_time))
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def row_encoder(row):
        norm_question, norm_ctxs = row['norm_question'], row['norm_ctx']
        query_encode_ids, query_len = query_encoder(query=norm_question, tokenizer=tokenizer)
        ################################################################################################################
        ctx_encodes = []
        for ctx_idx, content in enumerate(norm_ctxs):
            encode_tuple = context_encoder(content=content, tokenizer=tokenizer)
            ctx_encodes.append(encode_tuple)
        return query_encode_ids, ctx_encodes
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    start_time = time()
    encode_names = ['ques_encode', 'ctx_encode_list']
    data[encode_names] = data.swifter.apply(lambda row: pd.Series(row_encoder(row)), axis=1)
    print('Step 2: Tokenizing takes {:.4f} seconds'.format(time() - start_time))
    print('Number of data be processed = {}'.format(data.shape))
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def long_combination_encoder(row):
        query_encode, ctx_encode = row['ques_encode'], row['ctx_encode_list']
        doc_infor, sent_infor, seq_infor = test_context_merge_longer(query_encode_ids=query_encode,
                                                                              context_tuple_list=ctx_encode)
        doc_num, doc_len_array, doc_start_position, doc_end_position = doc_infor
        sent_num, sent_len_array, sent_start_position, sent_end_position, sent2doc_map_array, abs_sentIndoc_array, doc_sent_nums = sent_infor
        concat_ctx_array, token_num, global_attn_marker, token2sentID_map, answer_mask = seq_infor
        return doc_num, doc_len_array, doc_start_position, doc_end_position, \
               sent_num, sent_len_array, sent_start_position, sent_end_position, sent2doc_map_array, abs_sentIndoc_array, doc_sent_nums, \
               concat_ctx_array, token_num, global_attn_marker, token2sentID_map, answer_mask
    start_time = time()
    comb_res_col_names = ['doc_num', 'doc_len', 'doc_start', 'doc_end',
                          'sent_num', 'sent_len', 'sent_start', 'sent_end', 'sent2doc', 'sentIndoc', 'doc_sent_num',
                          'ctx_encode', 'ctx_len', 'global_attn', 'token2sent', 'ans_mask']
    data[comb_res_col_names] = \
        data.swifter.apply(lambda row: pd.Series(long_combination_encoder(row)), axis=1)
    print('Step 3: Combination takes {:.4f} seconds'.format(time() - start_time))
    print('Number of data be processed = {}'.format(data.shape))
    data_loader_res_columns = comb_res_col_names + ['_id']
    combined_data = data[data_loader_res_columns]
    norm_encode_col_names = normalized_names + encode_names + ['_id']
    ind_encoded_data = data[norm_encode_col_names]
    return data, combined_data, ind_encoded_data
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++