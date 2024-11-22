import pandas as pd
import numpy as np

import re

class Preprocessor:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    @property
    def mappers(self):
        return {
            "nan_to_value_mappers": {
                "accident": "None reported",
                "clean_title": "No"
            },
            "value_to_nan_mappers": {
                "fuel_type": "-",
            },
            "ftr_generator": {
                "engine": self.__get_new_features_from_engine
            }
        }

    def replace_nan_with_value(self, columns: list[str]) -> pd.DataFrame:
        for column in columns:
            self.data[column] = self.data[column].fillna(self.mappers["nan_to_value_mappers"][column])

        return self.data
    
    def replace_value_with_nan(self, columns: list[str]) -> pd.DataFrame:
        for column in columns:
            self.data[column].replace(self.mappers["value_to_nan_mappers"][column], np.nan, inplace=True)

        return self.data

    def fill_na_values(self, column: str, filling_type: str = "mode") -> pd.DataFrame:
        if filling_type == "mode":
            self.data[column].fillna(self.data[column].mode()[0], inplace=True)

            return self.data
        
        raise NotImplementedError
    
    def transform_ctg_to_num(self, categories: list[dict]):
        for column, mode in categories.items():
            if mode == "default":
                self.data[column] = self.data[column].astype("category").cat.codes
            elif mode == "one_hot":
                one_hot = pd.get_dummies(self.data[column], prefix=column)
                self.data = pd.concat([self.data.drop(columns=[column]), one_hot], axis=1)
            else:
                raise ValueError(f'Unknown mode: {mode}')

        return self.data

    def __get_new_features_from_engine(self, decription: str):
        horsepower_match = re.search(r'(\d+(\.\d+)?)HP', decription)
        horsepower = float(horsepower_match.group(1)) if horsepower_match else None

        engine_size_match = re.search(r'(\d+(\.\d+)?)L', decription)
        engine_size = float(engine_size_match.group(1)) if engine_size_match else None

        cylinders_match = re.search(r'(\d+) Cylinder', decription)
        cylinders = int(cylinders_match.group(1)) if cylinders_match else None

        fuel_match = re.search(r'\b(Gasoline|Diesel|Electric|Hybrid|Flex Fuel)\b', decription, re.IGNORECASE)
        fuel_type = fuel_match.group(0) if fuel_match else None
        if fuel_type == "Flex Fuel":
            fuel_type = "E85 Flex Fuel"

        return {
            'horsepower': horsepower,
            'engine_size': engine_size,
            'cylinders': cylinders,
            'fuel_type_from_engine': fuel_type
        }
    
    def create_new_features(self, column: str): 
        new_features = self.data[column].apply(self.mappers["ftr_generator"][column])

        new_features_df = pd.DataFrame(new_features.tolist())

        self.data = pd.concat([self.data, new_features_df], axis=1)

        return self.data
    




    

