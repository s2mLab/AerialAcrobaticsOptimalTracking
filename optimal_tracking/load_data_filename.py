def load_data_filename(subject, trial):
    if subject == 'DoCi':
        model_name = 'DoCi.s2mMod'
        if trial == '44_3':
            c3d_name = 'Do_44_mvtPrep_3.c3d'
            frames = range(4099, 4350)
    else:
        raise Exception(subject + ' is not a valid subject')

    data_filename = {
        'model': model_name,
        'c3d': c3d_name,
        'frames': frames,
    }

    return data_filename