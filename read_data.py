import pandas as pd 
def get_data():
    n = 100000
    df_05 = pd.read_parquet('df_05.parquet',  
                        filters=[('__index_level_0__', '<', n)]) 
    df_06 = pd.read_parquet('df_06.parquet',  
                        filters=[('__index_level_0__', '<', n)]) 
    df_l = pd.concat([df_05, df_06], axis=0, ignore_index=True)

    df_v = pd.read_parquet('video_stat.parquet',  
                        filters=[('__index_level_0__', '<', n)]) 

    video_id_mapping = {}

    def convert_video_ids(df):
        new_id = 1
        # Проходим по каждой строке DataFrame
        for index, row in df.iterrows():
            # Если video_id еще не встретился, добавляем его в словарь
            if row['video_id'] not in video_id_mapping:
                video_id_mapping[row['video_id']] = new_id
                new_id += 1

        # Заменяем старые video_id на новые в столбце 'video_id'
        df['video_id'] = df['video_id'].map(video_id_mapping)

        return df

    df_v = convert_video_ids(df_v)
    df_l = convert_video_ids(df_l)



    return df_l, df_v