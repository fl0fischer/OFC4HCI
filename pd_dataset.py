import numpy as np
import pandas as pd
import os

def datasetpreparation(dir_path=None, onset_threshold=1):
    """
    This function splits the Pointing Dynamics Dataset into individual trials
    and stores the associated information (start and end time step, start time step after removed reaction time)
    in a Pandas DataFrame.
    :param dir_path: local path to Mueller's PointingDynamicsDataset (str)
    :param onset_threshold: for each movement, we drop all frames before the velocity reaches onset_threshold percent
    of its maximum/minimum value (depending on the movement direction) for the first time
    :return: DataFrame consisting of all relevant information for trial identification
    """

    # provide local path to Mueller's PointingDynamicsDataset
    # (http://joergmueller.info/controlpointing/PointingDynamicsDataset.zip)
    if dir_path is None:
        dir_path = input("Insert the path to the PointingDynamicsDataset directory: ")
        dir_path = os.path.expanduser(dir_path)
    assert os.path.exists(dir_path)

    uservec = range(1, 12 + 1)
    directions = ('right', 'left')
    target = {(765, 255): 2, (1275, 425): 2, (765, 51): 4, (1275, 85): 4, (765, 12): 6, (1275, 20): 6, (765, 3): 8,
              (1275, 5): 8}

    numberOfTrials = 7732
    datalist = []
    for ii1 in uservec:
        for ii2 in directions:
            for ii3 in target.keys():
                (trialidxCURRENT, trialstartCURRENT, ndeltaCURRENT, nexttrialstartCURRENT) = readdataset(ii1, ii2, ii3[0], ii3[1],
                                                                                       onset_threshold, dir_path)
                datalist.append({'User': ii1, 'Direction': ii2, 'Distance': ii3[0], 'Width': ii3[1],
                                 'TrialIdx': trialidxCURRENT, 'N0': trialstartCURRENT, 'ndelta': ndeltaCURRENT, 'N1': nexttrialstartCURRENT})
    PointingDynamicDATA = pd.DataFrame(datalist)

    return PointingDynamicDATA


def readdataset(user, direction, distance, width, onset_threshold, dir_path):
    """
    This function extracts information on the chosen pointing task from the
    preprocessed data, namely the number and the indices of trials that are to be
    considered.
    :param user: number of test subject (required for filename)
    :param direction: movement direction; 'right': start from -0.1055, -'left': start from 0.1055
    :param distance: distance between target centers in pixel (required for filename)
    :param width: width of targets in pixel (required for filename)
    :param onset_threshold: reaction time threshold (for each movement, we drop all frames before the velocity reaches onset_threshold percent
    of its maximum/minimum value (depending on the movement direction) for the first time)
    :param dir_path: local path to Mueller's PointingDynamicsDataset (str)
    :return:
         - trialidx:           movement indices of currently considered trials
         - trialstart:         initial time step indices of considered trials
         - ndeltaarray:        initial time step indices of considered trials (after removing reaction times)
         - nexttrialstart:     initial time step indices of subsequent trials (i.e., initial time steps of trials after considered trials)
    """

    # provide local path to Mueller's PointingDynamicsDataset (http://joergmueller.info/controlpointing/PointingDynamicsDataset.zip)
    filenameread = f"{dir_path}/P{user}/Data0,-1,{distance},{width}.csv"

    DATA = pd.read_csv(filenameread)
    Ncomplete = len(DATA)
    dim = 1  # dimension of pointing task
    dt = np.mean(np.diff(DATA.time))  # seconds

    # NOTE: targets = ALL target switch time steps of the given dataset (i.e., initial time steps of all trials in the dataset + final time step)
    targets_diffs = [j - i for i, j in zip(DATA.target[:-1], DATA.target[1:])]
    targets_diffs_nonzero = [i for i, e in enumerate(targets_diffs) if e != 0]
    targets = np.array([0] + [x + 1 for x in targets_diffs_nonzero] + [Ncomplete - 1])  # '+1' used to define targets as first time steps with new target ("after click") [note shift in targets_diffs]
    if targets[-1] is targets[-2]:  # if target switches at time step N, indexing this target twice must be avoided
        targets = targets[:-1]

    # NOTE: trialidx = array containing respective indices of 'targets'
    # (starting time steps [time steps during trial]) that fit to chosen task [except last one!!!]
    if direction == 'right':  # only regard partial trajectories that fit to chosen task (and omit last entry to get only admissible starting points)
        trialidx = np.argwhere((DATA.target[targets[:-1]] == np.amax(DATA.target[targets[:-1]])).to_numpy()).flatten()
    elif direction == 'left':
        trialidx = np.argwhere((DATA.target[targets[:-1]] == np.amin(DATA.target[targets[:-1]])).to_numpy()).flatten()
    else:
        print('ERROR! Wrong task!\n')
        return 0

    # NOTE: to exclude trials where target was missed either at the beginning or at the end of the trial,
    # 'DATA.target' needs to be evaluated at *preceding* time step
    # (since at the click times stored in 'targets', DATA.target is already set to next target)
    if trialidx[0] == 0:   # NOTE: in the very first trial, starting position is assumed to equal respective "target" position -> consider only trialidx[1:]
        targets_minus1 = np.array([targets[i] - 1 for i in trialidx[1:]])
        targets_next_minus1 = np.array([targets[i+1] - 1 for i in trialidx[1:]])
        trialidxcopy=trialidx[1:]
        trialidx = np.append(trialidx[0], trialidxcopy[(np.absolute(
            DATA.y[targets[trialidxcopy]].to_numpy() - DATA.target[targets_minus1].to_numpy()) <= DATA.widthm[
                                           targets_minus1]).reset_index(drop=True) & (np.absolute(
            DATA.y[targets[trialidxcopy + 1]].to_numpy() - DATA.target[targets_next_minus1].to_numpy()) <=
                                                                                        DATA.widthm[targets_next_minus1]).reset_index(
            drop=True)])
    else:
        targets_minus1 = np.array([targets[i] - 1 for i in trialidx])
        targets_next_minus1 = np.array([targets[i+1] - 1 for i in trialidx])
        trialidx = trialidx[(np.absolute(
            DATA.y[targets[trialidx]].to_numpy() - DATA.target[targets_minus1].to_numpy()) <= DATA.widthm[
                                           targets_minus1]).reset_index(drop=True) & (np.absolute(
            DATA.y[targets[trialidx + 1]].to_numpy() - DATA.target[targets_next_minus1].to_numpy()) <=
                                                                                        DATA.widthm[targets_next_minus1]).reset_index(
            drop=True)]
    trialstart = targets[trialidx]
    nexttrialstart = targets[trialidx+1]

    ndelta = []  # initial time step indices of considered trials after removing reaction times
    ### VELOCITY ONSET CRITERION (RELATIVE) WITH ROLLING-WINDOW ACCELERATION SIGNUM CONSTRAINT:
    if direction == 'right':
        for i, j in zip(trialstart, nexttrialstart):
            condition_list = [(DATA.sgv[i:j] > (onset_threshold / 100) * np.max(DATA.sgv[i:j])),
                              ((DATA.sga[i:j]).rolling(window=pd.api.indexers.FixedForwardWindowIndexer(window_size=20)).min() > 0)]
            try:
                ndelta.append(DATA[i:j][np.bitwise_and.reduce(condition_list)].index[0])
            except IndexError:
                print(f"WARNING! Cannot satisfy the condition(s) {(user, distance, width, direction)} (TRIAL ID {i})!")
                print("TRIAL IS OMITTED.....")
                trialidx = trialidx[trialstart != i]
                trialstart = np.delete(trialstart, np.where(trialstart == i))
                nexttrialstart = np.delete(nexttrialstart, np.where(nexttrialstart == j))
    elif direction == 'left':
        for i, j in zip(trialstart, nexttrialstart):
            condition_list = [(DATA.sgv[i:j] < (onset_threshold / 100) * np.min(DATA.sgv[i:j])),
                              ((DATA.sga[i:j]).rolling(window=pd.api.indexers.FixedForwardWindowIndexer(window_size=20)).max() < 0)]
            try:
                ndelta.append(DATA[i:j][np.bitwise_and.reduce(condition_list)].index[0])
            except IndexError:
                print(f"WARNING! Cannot satisfy the condition(s) {(user, distance, width, direction)} (TRIAL ID {i})!")
                print("TRIAL IS OMITTED.....")
                trialidx = trialidx[trialstart != i]
                trialstart = np.delete(trialstart, np.where(trialstart == i))
                nexttrialstart = np.delete(nexttrialstart, np.where(nexttrialstart == j))
    else:
        print('ERROR! Wrong task!\n')
        return 0

    ndeltaarray = np.asarray(ndelta)

    return trialidx, trialstart, ndeltaarray, nexttrialstart
