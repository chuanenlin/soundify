import csv
import json
import statistics

def extract_answers(data):
  classify_ans = ''.join([key for key, val in data[0]['classify'].items() if val == True])
  sync_ans = ''.join([key for key, val in data[0]['sync'].items() if val == True])
  pan_ans = ''.join([key for key, val in data[0]['pan'].items() if val == True])
  vol_ans = ''.join([key for key, val in data[0]['volume'].items() if val == True])
  overall_ans = ''.join([key for key, val in data[0]['overall'].items() if val == True])
  return classify_ans, sync_ans, pan_ans, vol_ans, overall_ans

def ans_to_stat(ans):
  if ans == 'strongly agree':
    return 5
  elif ans == 'agree':
    return 4
  elif ans == 'neutral':
    return 3
  elif ans == 'disagree':
    return 2
  elif ans == 'strongly disagree':
    return 1

def group_ratings(filename, classify_dict, sync_dict, pan_dict, vol_dict, overall_dict, classify_ans, sync_ans, pan_ans, vol_ans, overall_ans):
  video_id = filename.split('-', 1)[1]
  if video_id in classify_dict:
    classify_dict[video_id].append(ans_to_stat(classify_ans))
    sync_dict[video_id].append(ans_to_stat(sync_ans))
    pan_dict[video_id].append(ans_to_stat(pan_ans))
    vol_dict[video_id].append(ans_to_stat(vol_ans))
    overall_dict[video_id].append(ans_to_stat(overall_ans))
  else:
    classify_dict[video_id] = [ans_to_stat(classify_ans)]
    sync_dict[video_id] = [ans_to_stat(sync_ans)]
    pan_dict[video_id] = [ans_to_stat(pan_ans)]
    vol_dict[video_id] = [ans_to_stat(vol_ans)]
    overall_dict[video_id] = [ans_to_stat(overall_ans)]
  return classify_dict, sync_dict, pan_dict, vol_dict, overall_dict

def mean_ratings(classify_dict, sync_dict, pan_dict, vol_dict, overall_dict):
  for key in classify_dict:    
    classify_dict[key] = statistics.mean(classify_dict[key])
  for key in sync_dict:    
    sync_dict[key] = statistics.mean(sync_dict[key])
  for key in pan_dict:    
    pan_dict[key] = statistics.mean(pan_dict[key])
  for key in vol_dict:    
    vol_dict[key] = statistics.mean(vol_dict[key])
  for key in overall_dict:    
    overall_dict[key] = statistics.mean(overall_dict[key])
  return classify_dict, sync_dict, pan_dict, vol_dict, overall_dict

def dict_to_arr(classify_dict, sync_dict, pan_dict, vol_dict, overall_dict):
  classify_arr = [classify_dict[key] for key in classify_dict]
  sync_arr = [sync_dict[key] for key in sync_dict]
  pan_arr = [pan_dict[key] for key in pan_dict]
  vol_arr = [vol_dict[key] for key in vol_dict]
  overall_arr = [overall_dict[key] for key in overall_dict]
  return classify_arr, sync_arr, pan_arr, vol_arr, overall_arr

def get_statistics(reader, system):
  classify_dict, sync_dict, pan_dict, vol_dict, overall_dict = {}, {}, {}, {}, {}
  for row in reader:
    filename = row[27]
    if filename.startswith(system):
      data = json.loads(row[28])
      if data[0]['video_played'] == 'false':
        continue
      if data[0]['task_time'] < 7:
        continue
      classify_ans, sync_ans, pan_ans, vol_ans, overall_ans = extract_answers(data)
      classify_dict, sync_dict, pan_dict, vol_dict, overall_dict = group_ratings(filename, classify_dict, sync_dict, pan_dict, vol_dict, overall_dict, classify_ans, sync_ans, pan_ans, vol_ans, overall_ans)
  classify_dict, sync_dict, pan_dict, vol_dict, overall_dict = mean_ratings(classify_dict, sync_dict, pan_dict, vol_dict, overall_dict)
  classify_arr, sync_arr, pan_arr, vol_arr, overall_arr = dict_to_arr(classify_dict, sync_dict, pan_dict, vol_dict, overall_dict)

  print(system)
  print('Mean')
  print(statistics.mean(classify_arr))
  print(statistics.mean(sync_arr))
  print(statistics.mean(pan_arr))
  print(statistics.mean(vol_arr))
  print(statistics.mean(overall_arr))
  print('SD')
  print(statistics.stdev(classify_arr))
  print(statistics.stdev(sync_arr))
  print(statistics.stdev(pan_arr))
  print(statistics.stdev(vol_arr))
  print(statistics.stdev(overall_arr))

with open('data.csv', 'r') as r:
  reader = csv.reader(r, delimiter=',')
  next(reader)
  get_statistics(reader, 'soundify')

with open('data.csv', 'r') as r:
  reader = csv.reader(r, delimiter=',')
  next(reader)
  get_statistics(reader, 'baseline')