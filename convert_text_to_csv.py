import gzip
import csv
import os
from typing import List

def extract_features(filename: str) -> None:
    """
    Read file from file name, extract features, then parse into csv file.
    """
    
    csv_fields = ['product/productId', 
                  'review/userId', 
                  'review/profileName', 
                  'review/helpfulness',
                  'review/score',
                  'review/time',
                  'review/summary',
                  'review/text']
    
    def helper(f):
        csv_values = []
        curr_review = []
        
        for linenumber, line in enumerate(f):
            if line == '\n':
                csv_values.append(curr_review)
                curr_review = []
                continue

            try:
                colon_idx = line.index(':')
                field, value = line[:colon_idx], line[colon_idx+1:].strip()
                curr_review.append(value)
                
            except:
                curr_review[-1] += '\n' + line
                print('Badly formatted line:')
                print(f'at {linenumber}: {line}', end='\n')
        
        return csv_values
        
        
    
    if filename.endswith('.gz'):
        with gzip.open(filename, 'rt', encoding='latin-1') as f:
            csv_values = helper(f)
    
    else:
        with open(filename, encoding='latin-1') as f:
            csv_values = helper(f)
            
    
    with open('finefoods.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file)

        csv_writer.writerow(csv_fields)

        csv_writer.writerows(csv_values)
        

  
        
        
if __name__ == '__main__':
    print("Functions to convert Amazon Finefoods text to csv")
    filename = input("Enter filename for features to extract:\n")
    
    extract_features(filename)