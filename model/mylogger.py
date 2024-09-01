import logging
import datetime

logdir = './logs/'

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    # formatter = logging.Formatter(
    #     "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    # )
    formatter = logging.Formatter(
    "[%(asctime)s] %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(logdir+filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    # sh = logging.StreamHandler()
    # sh.setFormatter(formatter)
    # logger.addHandler(sh)

    return logger
 
# 获取当前时间
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# logger = get_logger('AST_embedding.log')
logger = get_logger(f"APC_embedding_{current_time}.log")