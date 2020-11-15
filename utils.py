import os
import time
import logging
import datetime 

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def gen_checkpoint_id(args):
    timez   = datetime.datetime.now().strftime("%Y%m%d%H%M")
    checkpoint_id = "_".join([args.encoder_name, timez])
    return checkpoint_id

def get_logger(args):
    log_path = f"{args.checkpoint}/info/"

    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    train_instance_log_files = os.listdir(log_path)
    train_instance_count = len(train_instance_log_files)

    logging.basicConfig(
        filename=f'{args.checkpoint}/info/train_instance_{train_instance_count}_info.log', 
        filemode='w', 
        format="%(asctime)s | %(filename)15s | %(levelname)7s | %(funcName)10s | %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logger.info("-"*40)
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    logger.info("-"*40)\

    return logger

def checkpoint_count(checkpoint):
    _, folders, files = next(iter(os.walk(checkpoint)))
    files = list(filter(lambda x: "saved_checkpoint_" in x, files))
    # regex used to extract only integer elements from the list of files in the corresponding folder
    # this is to extract the most recent checkpoint in case of continuation of training
    checkpoints = map(lambda x: int(re.search(r"[0-9]{1,}", x).group()[0]), files)
    
    try:
        last_checkpoint = sorted(checkpoints)[-1]
    except IndexError:
        last_checkpoint = 0
    return last_checkpoint


