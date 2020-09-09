from download_data import download_data
from convert_text_to_csv import extract_features
from model import get_data, train

# Driver program

if __name__ == '__main__':
    download_data()

    print('Data downloaded.')

    extract_features('finefoods.txt.gz')

    print('Data parsed and extracted.')

    data = get_data('finefoods.csv')

    train(data)