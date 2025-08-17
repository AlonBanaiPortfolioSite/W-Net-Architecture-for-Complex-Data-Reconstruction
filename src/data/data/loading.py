import wfdb


def loading_and_filtering_(file_path,database_path,min_duration,required_signal,sampling_freq):
    '''
    Loads all the records listed in the file path that:
      - include all required signals,
      - have duration >= min_duration (in minutes),
      - have the specified sampling frequency.

    Inputs:
        file_path (str): Path to a .txt file listing record names (one per line)
        min_duration (float): Minimum required duration in minutes
        required_signal (list of str): List of required signal names (e.g., ["PLETH", "II"])
        sampling_freq (float): Required sampling frequency in Hz

    Returns:
        valid_records_list (list of str): Paths to valid records
    '''
    valid_records_list = []
    with open(file_path, 'r') as file:
        record_list=sorted(file.readlines())#getting a list of all records alphabeticly
    for record in record_list:
        #loading records
        record_path = f"{database_path}/{record.strip()}"
        record_header = wfdb.rdheader(record_path, rd_segments=True)
        #filtering only records that folow the detrmined cratrias
        valid_freq = (record_header.fs == sampling_freq)#filter only records with the right sampling fraquncy
        valid_length =(record_header.sig_len / (record_header.fs * 60) >= min_duration)#filter only records that are at least min_duration
        valid_names = True
        #filtering based on the prsense of the relvant signals
        valid_names = all(signal_type in record_header.sig_name for signal_type in required_signal)
        

        if valid_length and valid_names and valid_freq:
            valid_records_list.append(record_path)

    return valid_records_list
            