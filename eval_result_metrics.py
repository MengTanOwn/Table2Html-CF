from evaluate import load
import json
import os
import pandas as pd
from bs4 import BeautifulSoup
from src.metric import TEDS


# 加载评估指标
bleu_metric = load("bleu")

# 加载ROUGE评估指标
rouge = load("rouge")


def get_structure(content):
    try:
        soup_pred = BeautifulSoup(content, 'html.parser')
        for td in soup_pred.find_all('td'):
            # 清空所有文本内容
            td.string.replace_with('')
        cleaned_html_pred = soup_pred.prettify()
        return cleaned_html_pred
    except:
        return ''

def coupute_text_and_structure_acc(references,predictions):
    acc_count = 0
    struct_count = 0
    teds = TEDS()
    teds_score = 0
    for i,j in zip(references,predictions):
    
        # Evaluate
        try:
            teds_item_sroce = teds.evaluate(f'<html>{j}</html>', f'<html>{i}</html>')
            print('teds_item_sroce:',teds_item_sroce)
            teds_score += teds_item_sroce
        except:
            print(i,'\n',j)
            pass
        # if i==j:
        if teds_item_sroce>=0.95:
            acc_count += 1
        i_s = get_structure(i)
        j_s = get_structure(j)
        # if i_s == j_s:
        #     struct_count += 1
        
        try:
            struct_count_score = teds.evaluate(f'<html>{j_s}</html>', f'<html>{i_s}</html>')
            print('teds_strcuct_sroce:',struct_count_score)
            struct_count += struct_count_score
        except:
            print(i_s,'\n',j_s)
    return round(acc_count/len(references),4),round(struct_count/len(references),4),round(teds_score/len(references),4)

def compute_metrics(filename):
    references = []
    predictions = []
    row ={}
    row['model-name'] = os.path.basename(filename)[:-6]
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            references.append(record['label'])
            predictions.append(record['predict'])
    # 计算BLEU分数
    for order in [1,2,3,4]:
        results = bleu_metric.compute(predictions=predictions, references=references,max_order=order)
        row[f'BLEU-{order}']=round(results['bleu'],4)

    print(f"BLEU score: {results['bleu']}")
    # 计算ROUGE分数
    results_rouge = rouge.compute(predictions=predictions, 
                                  references=references,
                                  rouge_types=['rouge1','rouge2','rouge3','rouge4', 'rougeL'])
    # 输出结果
    print('rouge',results_rouge)
    
    for key ,value in results_rouge.items():
        row[key] = round(value,4)
    acc , acc_structure,teds_score = coupute_text_and_structure_acc(references,predictions)
    row['TEDS score'] = teds_score
    row['absolutely accurate'] = acc
    row['structure accurate'] = acc_structure
    # print(teds_score)
    return row

filename_list = ['/mnt/localdisk/tanm/nltk_data/backup_llm/swift/baseline-results/baseline_glm4v_9b_chat_predict_wo_context.jsonl',
                 '/mnt/localdisk/tanm/nltk_data/backup_llm/swift/baseline-results/baseline_internvl_chat_v1_5_predict_wo_context.jsonl',
                 '/mnt/localdisk/tanm/nltk_data/backup_llm/swift/baseline-results/baseline_internvl2_8b_predict_wo_context.jsonl',
                 '/mnt/localdisk/tanm/nltk_data/backup_llm/swift/baseline-results/baseline_minicpm_v_v2_5_chat_predict_wo_context.jsonl',
                 '/mnt/localdisk/tanm/nltk_data/backup_llm/swift/baseline-results/baseline_qwen_vl_chat_predict_wo_context.jsonl',
                 '/mnt/localdisk/tanm/nltk_data/backup_llm/swift/baseline-results/baseline_yi_vl_6b_chat_predict_wo_context.jsonl',
                 '/mnt/localdisk/tanm/nltk_data/backup_llm/LLaMA-FIN/saves/llava_hf_V2_2_without_contex_prompt/lora_rank32/predict/random_template/generated_predictions.jsonl'
                 ]    
data = []
for filename in filename_list:
    # filename = 'fin_table2html_generated.jsonl'
    row = compute_metrics(filename)
    data.append(row)

# 创建DataFrame
df = pd.DataFrame(data)

result_save_file = 'baseline_analysis2_and_wo_context_prompt.csv'
# 将DataFrame写入CSV文件
df.to_csv(result_save_file, index=False,)
print(f"Metrics saved to {result_save_file}")