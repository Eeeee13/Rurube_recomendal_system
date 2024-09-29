from enum import Enum
from T import T
from star import star
from collab import collab
from similar import similar
from populariry import popularity
# from check_title import Check_title
from read_data import get_data

df_l, df_v = get_data()
amount = 100
black_list = set()
like = set()
dislike = set()
user_id = ''

class SM(Enum):
    HOME_Z = 1
    WATCH = 2
    HOME_N = 3


    

def start():
    videos = []
    time_videos = T(df_l, df_v, amount)
    popular_videos = popularity(df_v=df_v, N = amount)
    star_t_v = star(time_videos, 10)
    star_pop = star(popular_videos, 10)
    videos.append(star_pop, star_t_v)
    black_list.add(videos)
    return videos

def work():
    videos = []
    sim_videos = similar(df_v)
    popular_videos = popularity(df_v=df_v, N = amount)
    collab_videos = collab(df_l, df_v, 2, like, dislike, user_id)
    star_s_v = star(sim_videos, 6)
    star_p_v = star(popular_videos, 2)
    videos.append(collab_videos,star_s_v,star_p_v)
    
    black_list.add(videos)
    return videos
    

match SM:
    case SM.HOME_Z:
        start()


    case SM.WATCH:
        pass


    case SM.HOME_N:
        pass

