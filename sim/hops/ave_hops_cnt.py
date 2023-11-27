
import re
if __name__ == '__main__':
    file = open('hops_bert_large.txt', 'r')
    #file = open('hops_resnet50.txt', 'r')
    try:
        while True:
            text_line = file.readlines()

            if text_line:
                #print(type(text_line), text_line)
                cnt_comm=0
                cnt_hops=0
                for line in text_line:
                    line=line.strip()
                    #print(line)
                    pattern = r'\[(\d+(?:\s*,\s*\d+)*)\]'
                    matches = re.findall(pattern, line)
                    if matches:
                        #print(matches)
                        cnt_comm+=1
                        numbers = matches[0].split(',')
                        cnt_hops+=len(numbers)
                    else:
                        continue
                print('ave_hops_cnt={:.3f}\n'.format(cnt_hops/cnt_comm))
            else: 
                break
    except Exception as e:
        print('read error', e)
    finally:
        file.close()

    
