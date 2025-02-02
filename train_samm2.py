import os

import numpy as np
from sklearn.svm import SVC
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping

from models import VGG_16, temporal_module, VGG_16_4_channels, convolutional_autoencoder
from samm_utilitis import get_subfolders_num_crossdb, Read_Input_Images_SAMM_CASME
from utilities import Read_Input_Images, get_subfolders_num, data_loader_with_LOSO, label_matching, duplicate_channel
from utilities import loading_casme_table, ignore_casme_samples, LossHistory
from utilities import loading_samm_table, loading_casme_objective_table, filter_objective_samples


def train_samm(batch_size, spatial_epochs, temporal_epochs, train_id, dB, spatial_size, flag, tensorboard):
    ############## Path Preparation ######################
    root_db_path = "/media/ice/OS/Datasets/"
    workplace = root_db_path + dB + "/"
    inputDir = root_db_path + dB + "/" + dB + "/"
    ######################################################
    classes = 5
    if dB == 'CASME2_TIM':
        table = loading_casme_table(workplace + 'CASME2-ObjectiveClasses.xlsx')
        listOfIgnoredSamples, IgnoredSamples_index = ignore_casme_samples(inputDir)

        ############## Variables ###################
        r = w = spatial_size
        subjects = 2
        n_exp = 5
        # VidPerSubject = get_subfolders_num(inputDir, IgnoredSamples_index)
        listOfIgnoredSamples = []
        VidPerSubject = [2, 1]
        timesteps_TIM = 10
        data_dim = r * w
        pad_sequence = 10
        channel = 3
        ############################################

        os.remove(workplace + "Classification/CASME2_TIM_label.txt")



    elif dB == 'CASME2_Optical':
        table = loading_casme_table(workplace + 'CASME2-ObjectiveClasses.xlsx')
        listOfIgnoredSamples, IgnoredSamples_index = ignore_casme_samples(inputDir)

        ############## Variables ###################
        r = w = spatial_size
        subjects = 26
        n_exp = 5
        VidPerSubject = get_subfolders_num(inputDir, IgnoredSamples_index)
        timesteps_TIM = 9
        data_dim = r * w
        pad_sequence = 9
        channel = 3
    ############################################

    # os.remove(workplace + "Classification/CASME2_TIM_label.txt")

    elif dB == 'SAMM_TIM10':
        table, table_objective = loading_samm_table(root_db_path, dB)
        listOfIgnoredSamples = []
        IgnoredSamples_index = np.empty([0])

        ################# Variables #############################
        r = w = spatial_size
        subjects = 29
        n_exp = 8
        VidPerSubject = get_subfolders_num(inputDir, IgnoredSamples_index)
        timesteps_TIM = 10
        data_dim = r * w
        pad_sequence = 10
        channel = 3
        classes = 8
    #########################################################

    elif dB == 'SAMM_CASME_Strain':
        # total amount of videos 253
        table, table_objective = loading_samm_table(root_db_path, dB)
        table = table_objective
        table2 = loading_casme_objective_table(root_db_path, dB)

        # merge samm and casme tables
        table = np.concatenate((table, table2), axis=1)

        # print(table.shape)

        # listOfIgnoredSamples, IgnoredSamples_index, sub_items = ignore_casme_samples(inputDir)
        listOfIgnoredSamples = []
        IgnoredSamples_index = np.empty([0])
        sub_items = np.empty([0])
        list_samples = filter_objective_samples(table)

        r = w = spatial_size
        subjects = 47  # some subjects were removed because of objective classes and ignore samples: 47
        n_exp = 5
        # TODO:
        # 1) Further decrease the video amount, the one with objective classes >= 6
        # list samples: samples with wanted objective class
        VidPerSubject, list_samples = get_subfolders_num_crossdb(inputDir, IgnoredSamples_index, sub_items, table,
                                                                 list_samples)

        # print(VidPerSubject)
        # print(len(VidPerSubject))
        # print(sum(VidPerSubject))
        timesteps_TIM = 9
        data_dim = r * w
        channel = 3
        classes = 5
        if os.path.isfile(workplace + "Classification/SAMM_CASME_Optical_label.txt"):
            os.remove(workplace + "Classification/SAMM_CASME_Optical_label.txt")
    ##################### Variables ######################

    ######################################################

    ############## Flags ####################
    tensorboard_flag = tensorboard
    resizedFlag = 1
    train_spatial_flag = 0
    train_temporal_flag = 0
    svm_flag = 0
    finetuning_flag = 0
    cam_visualizer_flag = 0
    channel_flag = 0

    if flag == 'st':
        train_spatial_flag = 1
        train_temporal_flag = 1
        finetuning_flag = 1
    elif flag == 's':
        train_spatial_flag = 1
        finetuning_flag = 1
    elif flag == 't':
        train_temporal_flag = 1
    elif flag == 'nofine':
        svm_flag = 1
    elif flag == 'scratch':
        train_spatial_flag = 1
        train_temporal_flag = 1
    elif flag == 'st4':
        train_spatial_flag = 1
        train_temporal_flag = 1
        channel_flag = 1
    elif flag == 'st7':
        train_spatial_flag = 1
        train_temporal_flag = 1
        channel_flag = 2
    #########################################

    ############ Reading Images and Labels ################

    SubperdB = Read_Input_Images_SAMM_CASME(inputDir, list_samples, listOfIgnoredSamples, dB, resizedFlag, table,
                                            workplace, spatial_size, channel)
    print("Loaded Images into the tray...")
    labelperSub = label_matching(workplace, dB, subjects, VidPerSubject)
    print("Loaded Labels into the tray...")

    if channel_flag == 1:
        SubperdB_strain = Read_Input_Images(inputDir, listOfIgnoredSamples, 'CASME2_Strain_TIM10', resizedFlag, table,
                                            workplace, spatial_size, 1)
    elif channel_flag == 2:
        SubperdB_strain = Read_Input_Images(inputDir, listOfIgnoredSamples, 'CASME2_Strain_TIM10', resizedFlag, table,
                                            workplace, spatial_size, 1)
        SubperdB_gray = Read_Input_Images(inputDir, listOfIgnoredSamples, 'CASME2_TIM', resizedFlag, table, workplace,
                                          spatial_size, 3)
    #######################################################

    ########### Model Configurations #######################
    sgd = optimizers.SGD(lr=0.0001, decay=1e-7, momentum=0.9, nesterov=True)
    adam = optimizers.Adam(lr=0.00001, decay=0.000001)
    adam2 = optimizers.Adam(lr=0.00075, decay=0.0001)

    # Different Conditions for Temporal Learning ONLY
    if train_spatial_flag == 0 and train_temporal_flag == 1 and dB != 'CASME2_Optical':
        data_dim = spatial_size * spatial_size
    elif train_spatial_flag == 0 and train_temporal_flag == 1 and dB == 'CASME2_Optical':
        data_dim = spatial_size * spatial_size * 3
    else:
        data_dim = 4096

    ########################################################

    ########### Training Process ############

    # total confusion matrix to be used in the computation of f1 score
    tot_mat = np.zeros((n_exp, n_exp))

    # model checkpoint
    spatial_weights_name = 'vgg_spatial_' + str(train_id) + '_casme2_'
    temporal_weights_name = 'temporal_ID_' + str(train_id) + '_casme2_'
    history = LossHistory()
    stopping = EarlyStopping(monitor='loss', min_delta=0, mode='min')

    for sub in range(subjects):
        ############### Reinitialization & weights reset of models ########################

        vgg_model_cam = VGG_16(spatial_size=spatial_size, classes=classes, weights_path='VGG_Face_Deep_16.h5')

        temporal_model = temporal_module(data_dim=data_dim, classes=classes, timesteps_TIM=timesteps_TIM)
        temporal_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])

        conv_ae = convolutional_autoencoder(spatial_size=spatial_size, classes=classes)
        conv_ae.compile(loss='binary_crossentropy', optimizer=adam)

        if channel_flag == 1 or channel_flag == 2:
            vgg_model = VGG_16_4_channels(classes=classes, spatial_size=spatial_size)
            vgg_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])
        else:
            vgg_model = VGG_16(spatial_size=spatial_size, classes=classes, weights_path='VGG_Face_Deep_16.h5')
            vgg_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])

        svm_classifier = SVC(kernel='linear', C=1)
        ####################################################################################

        ############ for tensorboard ###############
        if tensorboard_flag == 1:
            cat_path = tensorboard_path + str(sub) + "/"
            os.mkdir(cat_path)
            tbCallBack = keras.callbacks.TensorBoard(log_dir=cat_path, write_graph=True)

            cat_path2 = tensorboard_path + str(sub) + "spat/"
            os.mkdir(cat_path2)
            tbCallBack2 = keras.callbacks.TensorBoard(log_dir=cat_path2, write_graph=True)
        #############################################

        image_label_mapping = np.empty([0])

        Train_X, Train_Y, Test_X, Test_Y, Test_Y_gt = data_loader_with_LOSO(sub, SubperdB, labelperSub, subjects,
                                                                            classes)

        # Rearrange Training labels into a vector of images, breaking sequence
        Train_X_spatial = Train_X.reshape(Train_X.shape[0] * timesteps_TIM, r, w, channel)
        Test_X_spatial = Test_X.reshape(Test_X.shape[0] * timesteps_TIM, r, w, channel)

        # Special Loading for 4-Channel
        if channel_flag == 1:
            Train_X_Strain, _, Test_X_Strain, _, _ = data_loader_with_LOSO(sub, SubperdB_strain, labelperSub, subjects,
                                                                           classes)
            Train_X_Strain = Train_X_Strain.reshape(Train_X_Strain.shape[0] * timesteps_TIM, r, w, 1)
            Test_X_Strain = Test_X_Strain.reshape(Test_X.shape[0] * timesteps_TIM, r, w, 1)

            # Concatenate Train X & Train_X_Strain
            Train_X_spatial = np.concatenate((Train_X_spatial, Train_X_Strain), axis=3)
            Test_X_spatial = np.concatenate((Test_X_spatial, Test_X_Strain), axis=3)

            channel = 4

        elif channel_flag == 2:
            Train_X_Strain, _, Test_X_Strain, _, _ = data_loader_with_LOSO(sub, SubperdB_strain, labelperSub, subjects,
                                                                           classes)
            Train_X_gray, _, Test_X_gray, _, _ = data_loader_with_LOSO(sub, SubperdB_gray, labelperSub, subjects)
            Train_X_Strain = Train_X_Strain.reshape(Train_X_Strain.shape[0] * timesteps_TIM, r, w, 1)
            Test_X_Strain = Test_X_Strain.reshape(Test_X_Strain.shape[0] * timesteps_TIM, r, w, 1)
            Train_X_gray = Train_X_gray.reshape(Train_X_gray.shape[0] * timesteps_TIM, r, w, 3)
            Test_X_gray = Test_X_gray.reshape(Test_X_gray.shape[0] * timesteps_TIM, r, w, 3)

            # Concatenate Train_X_Strain & Train_X & Train_X_gray
            Train_X_spatial = np.concatenate((Train_X_spatial, Train_X_Strain, Train_X_gray), axis=3)
            Test_X_spatial = np.concatenate((Test_X_spatial, Test_X_Strain, Test_X_gray), axis=3)
            channel = 7

        if channel == 1:
            # Duplicate channel of input image
            Train_X_spatial = duplicate_channel(Train_X_spatial)
            Test_X_spatial = duplicate_channel(Test_X_spatial)

        # Extend Y labels 10 fold, so that all images have labels
        Train_Y_spatial = np.repeat(Train_Y, timesteps_TIM, axis=0)
        Test_Y_spatial = np.repeat(Test_Y, timesteps_TIM, axis=0)

        # print ("Train_X_shape: " + str(np.shape(Train_X_spatial)))
        # print ("Train_Y_shape: " + str(np.shape(Train_Y_spatial)))
        # print ("Test_X_shape: " + str(np.shape(Test_X_spatial)))
        # print ("Test_Y_shape: " + str(np.shape(Test_Y_spatial)))
        # print(Train_X_spatial)
        ##################### Training & Testing #########################

        X = Train_X_spatial.reshape(Train_X_spatial.shape[0], channel, r, w)
        y = Train_Y_spatial.reshape(Train_Y_spatial.shape[0], classes)
        normalized_X = X.astype('float32') / 255.

        test_X = Test_X_spatial.reshape(Test_X_spatial.shape[0], channel, r, w)
        test_y = Test_Y_spatial.reshape(Test_Y_spatial.shape[0], classes)
        normalized_test_X = test_X.astype('float32') / 255.

        print(X.shape)

        ###### conv weights must be freezed for transfer learning ######
        if finetuning_flag == 1:
            for layer in vgg_model.layers[:33]:
                layer.trainable = False

        if train_spatial_flag == 1 and train_temporal_flag == 1:
            # Autoencoder features
            # conv_ae.fit(normalized_X, normalized_X, batch_size=batch_size, epochs=spatial_epochs, shuffle=True)

            # Spatial Training
            if tensorboard_flag == 1:
                vgg_model.fit(X, y, batch_size=batch_size, epochs=spatial_epochs, shuffle=True, callbacks=[tbCallBack2])
            else:
                vgg_model.fit(X, y, batch_size=batch_size, epochs=spatial_epochs, shuffle=True,
                              callbacks=[history, stopping])

            # record f1 and loss
            file_loss = open(workplace + 'Classification/' + 'Result/' + dB + '/loss_' + str(train_id) + '.txt', 'a')
            file_loss.write(str(history.losses) + "\n")
            file_loss.close()

            file_loss = open(workplace + 'Classification/' + 'Result/' + dB + '/accuracy_' + str(train_id) + '.txt',
                             'a')
            file_loss.write(str(history.accuracy) + "\n")
            file_loss.close()

            vgg_model.save_weights(spatial_weights_name + str(sub) + ".h5")
