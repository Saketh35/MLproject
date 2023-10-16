import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig():
    train_path_data: str=os.path.join('artifacts','train.csv')
    test_path_data: str=os.path.join('artifacts','test.csv')
    raw_path_data: str=os.path.join('artifacts','data.csv')

class DataIngestion():
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the dataset as Data Frame')
            os.makedirs(os.path.dirname(self.ingestion_config.train_path_data),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_path_data,index=False,header=True)

            logging.info('Train test split initiated')
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=43)
            train_set.to_csv(self.ingestion_config.train_path_data,index=False,header=True)
            test_set.to_csv(self.ingestion_config.train_path_data,index=False,header=True)

            logging.info('Ingestion of the data is completed')
            return(
                self.ingestion_config.train_path_data,
                self.ingestion_config.test_path_data
            )

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()