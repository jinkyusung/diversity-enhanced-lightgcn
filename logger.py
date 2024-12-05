import config


def console(arg) -> None:
    msg = str(arg) + '\n'
    with open(config.log_file, 'a') as fp:
        fp.write(msg)
    return