

def clean_contract_data(train, year, month, offset):

    front_month_list = [1,3,5,7,8,9,10,12,
                    1,3,5,7,8,9,10,12,
                    1,3,5,7,8,9,10,12]

    back_month_list = [11,12,2,4,6,7,8,9,
                   11,12,2,4,6,7,8,9,
                   11,12,2,4,6,7,8,9]
    
    start_contract_in_list = 7
    
    if month =='Jan':
        month_in_number = 1
    elif month=='Feb':
        month_in_number = 2
    elif month=='Mar':
        month_in_number = 3
    elif month=='Apr':
        month_in_number = 4
    elif month=='May':
        month_in_number = 5
    elif month=='Jun':
        month_in_number = 6
    elif month=='Jul':
        month_in_number = 7
    elif month=='Aug':
        month_in_number = 8
    elif month=='Sep':
        month_in_number = 9
    elif month=='Oct':
        month_in_number = 10
    elif month=='Nov':
        month_in_number = 11
    elif month=='Dec':
        month_in_number = 12

    if (month_in_number - offset) >= 8: 
        year_actual_front = 2000+year

    elif (month_in_number - offset) < 8: 
        year_actual_front = 2000 + year - 1

    if (month_in_number - offset) >= 10: 
        year_actual_back = 2000+year

    elif (month_in_number - offset) < 10:
        year_actual_back = 2000 + year - 1

    back_month  = back_month_list[month_in_number-offset]

    front_month = front_month_list[month_in_number-offset]

    if back_month <= 9:
        back_separator = '-0'
    elif back_month >= 10:
        back_separator = '-'

    if front_month <= 9:
        front_separator = '-0'
    elif front_month >= 10:
        front_separator = '-'

    if month=='Jan':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[start_contract_in_list+1-offset]) + '-' + str(10)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[start_contract_in_list+1-offset]) + '-0' + str(1)

    elif month=='Mar':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[start_contract_in_list+2-offset]) + '-' + str(10)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[start_contract_in_list+2-offset]) + '-0' + str(1)

    elif month=='May':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[start_contract_in_list+3-offset]) + '-' + str(10)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[start_contract_in_list+3-offset]) + '-0' + str(1)

    elif month=='Jul':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[start_contract_in_list+4-offset]) + '-' + str(10)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[start_contract_in_list+4-offset]) + '-0' + str(1)

    elif month=='Aug':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[start_contract_in_list+5-offset]) + '-' + str(10)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[start_contract_in_list+5-offset]) + '-0' + str(1)

    elif month=='Sep':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[start_contract_in_list+6-offset]) + '-' + str(10)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[start_contract_in_list+6-offset]) + '-0' + str(1)

    elif month=='Oct':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[start_contract_in_list+7-offset]) + '-' + str(10)
        end_date = str(year_actual_front) +front_separator +str(front_month_list[start_contract_in_list+7-offset]) + '-0' + str(1)

    elif month=='Dec':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[start_contract_in_list+8-offset]) + '-' + str(10)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[start_contract_in_list+8-offset]) + '-0' + str(1)

    # print(start_date, end_date)

    start_date = pd.to_datetime(start_date, infer_datetime_format=True)
    end_date = pd.to_datetime(end_date, infer_datetime_format=True)
    data_mask = (train.index < end_date) & (train.index >= start_date)

    train = train.loc[data_mask]

    return train


def clean_backtest_data(train, tradeLog, year, month, month_in_number, offset):
    front_month_list = [1,3,5,7,8,9,10,12,
                    1,3,5,7,8,9,10,12,
                    1,3,5,7,8,9,10,12]

    back_month_list = [12,1,3,5,7,8,9,10,
                   12,1,3,5,7,8,9,10,
                   12,1,3,5,7,8,9,10]
    
    start_contract_in_list = 7
    
    if month =='Jan':
        month_in_number = 1
    elif month=='Feb':
        month_in_number = 2
    elif month=='Mar':
        month_in_number = 3
    elif month=='Apr':
        month_in_number = 4
    elif month=='May':
        month_in_number = 5
    elif month=='Jun':
        month_in_number = 6
    elif month=='Jul':
        month_in_number = 7
    elif month=='Aug':
        month_in_number = 8
    elif month=='Sep':
        month_in_number = 9
    elif month=='Oct':
        month_in_number = 10
    elif month=='Nov':
        month_in_number = 11
    elif month=='Dec':
        month_in_number = 12

    if (month_in_number - offset) >= 8: 
        year_actual_front = 2000+year

    elif (month_in_number - offset) < 8: 
        year_actual_front = 2000 + year - 1

    if (month_in_number - offset) >= 9: 
        year_actual_back = 2000+year

    elif (month_in_number - offset) < 9:
        year_actual_back = 2000 + year - 1

    back_month  = back_month_list[month_in_number-offset]

    front_month = front_month_list[month_in_number-offset]

    if back_month <= 9:
        back_separator = '-0'
    elif back_month >= 10:
        back_separator = '-'

    if front_month <= 9:
        front_separator = '-0'
    elif front_month >= 10:
        front_separator = '-'

    if month=='Jan':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[start_contract_in_list+1-offset]) + '-' + str(10)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[start_contract_in_list+1-offset]) + '-0' + str(1)

    elif month=='Mar':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[start_contract_in_list+2-offset]) + '-' + str(10)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[start_contract_in_list+2-offset]) + '-0' + str(1)

    elif month=='May':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[start_contract_in_list+3-offset]) + '-' + str(10)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[start_contract_in_list+3-offset]) + '-0' + str(1)

    elif month=='Jul':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[start_contract_in_list+4-offset]) + '-' + str(10)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[start_contract_in_list+4-offset]) + '-0' + str(1)

    elif month=='Aug':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[start_contract_in_list+5-offset]) + '-' + str(10)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[start_contract_in_list+5-offset]) + '-0' + str(1)

    elif month=='Sep':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[start_contract_in_list+6-offset]) + '-' + str(10)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[start_contract_in_list+6-offset]) + '-0' + str(1)

    elif month=='Oct':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[start_contract_in_list+7-offset]) + '-' + str(10)
        end_date = str(year_actual_front) +front_separator +str(front_month_list[start_contract_in_list+7-offset]) + '-0' + str(1)

    elif month=='Dec':
        start_date = str(year_actual_back) + back_separator +str(back_month_list[start_contract_in_list+8-offset]) + '-' + str(10)
        end_date = str(year_actual_front) + front_separator +str(front_month_list[start_contract_in_list+8-offset]) + '-0' + str(1)


    # print(start_date, end_date)

    start_date = pd.to_datetime(start_date, infer_datetime_format=True)
    end_date = pd.to_datetime(end_date, infer_datetime_format=True)

    trade_mask = (tradeLog['Entry Time'] <= end_date) & (tradeLog['Entry Time'] >= start_date)
    data_mask = (train.index < end_date) & (train.index >= start_date)

    revised_tradeLog = tradeLog.loc[trade_mask]
    train = train.loc[data_mask]

    return train, revised_tradeLog
