import datetime


def date_uid():
    """Generate a unique id based on date.
    Returns:
        str: Return uid string, e.g. '20171122171307111552'.
    """
    return str(datetime.datetime.now()).replace(
        '-', '').replace(
            ' ', '').replace(
                ':', '').replace('.', '')
