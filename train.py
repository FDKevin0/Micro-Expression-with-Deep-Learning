import gc
import os

import numpy as np
import tensorflow.keras
from sklearn.metrics import confusion_matrix
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping

from evaluationmatrix import fpr, weighted_average_recall, unweighted_average_recall
from list_databases import load_db, restructure_data
from models import VGG_16, temporal_module, VGG_16_4_channels
from samm_utilitis import Read_Input_Images_SAMM_CASME
from utilities import Read_Input_Images, label_matching
from utilities import gpu_observer
from utilities import record_loss_accuracy, record_weights, record_scores, LossHistory  # recording scripts


# import pydot, graphviz


def train(batch_size, spatial_epochs, temporal_epochs, train_id, list_dB, spatial_size, flag, objective_flag,
          tensorboard):
    ############## Path Preparation ######################
    root_db_path = "./Datasets/"
    tensorboard_path = root_db_path + "tensorboard/"
    if not os.path.isdir(root_db_path + 'Weights/' + str(train_id)):
        os.mkdir(root_db_path + 'Weights/' + str(train_id))

    ######################################################

    ############## Variables ###################
    dB = list_dB[0]
    r, w, subjects, samples, n_exp, vid_per_subject, timesteps_TIM, data_dim, channel, table, list_of_ignored_samples, db_home, db_images, cross_db_flag = load_db(
        root_db_path, list_dB, spatial_size, objective_flag)

    # avoid confusion
    if cross_db_flag == 1:
        list_samples = list_of_ignored_samples

    # total confusion matrix to be used in the computation of f1 score
    tot_mat = np.zeros((n_exp, n_exp))

    history = LossHistory()
    stopping = EarlyStopping(monitor='loss', min_delta=0, mode='min')

    ############################################

    ############## Flags ####################
    tensorboard_flag = tensorboard
    resizedFlag = 1
    train_spatial_flag = 0
    train_temporal_flag = 0
    svm_flag = 0
    fine_tuning_flag = 0
    cam_visualizer_flag = 0
    channel_flag = 0

    if flag == 'st':
        train_spatial_flag = 1
        train_temporal_flag = 1
        fine_tuning_flag = 1
    elif flag == 's':
        train_spatial_flag = 1
        fine_tuning_flag = 1
    elif flag == 't':
        train_temporal_flag = 1
    elif flag == 'nofine':
        svm_flag = 1
    elif flag == 'scratch':
        train_spatial_flag = 1
        train_temporal_flag = 1
    elif flag == 'st4se' or flag == 'st4se_cde':
        train_spatial_flag = 1
        train_temporal_flag = 1
        channel_flag = 1
    elif flag == 'st7se' or flag == 'st7se_cde':
        train_spatial_flag = 1
        train_temporal_flag = 1
        channel_flag = 2
    elif flag == 'st4te' or flag == 'st4te_cde':
        train_spatial_flag = 1
        train_temporal_flag = 1
        channel_flag = 3
    elif flag == 'st7te' or flag == 'st7te_cde':
        train_spatial_flag = 1
        train_temporal_flag = 1
        channel_flag = 4

    #########################################

    ############ Reading Images and Labels ################
    if cross_db_flag == 1:
        SubperdB = Read_Input_Images_SAMM_CASME(db_images, list_samples, list_of_ignored_samples, dB, resizedFlag, table,
                                                db_home, spatial_size, channel)
    else:
        SubperdB = Read_Input_Images(db_images, list_of_ignored_samples, dB, resizedFlag, table, db_home, spatial_size,
                                     channel, objective_flag)

    label_per_sub = label_matching(db_home, dB, subjects, vid_per_subject)
    print("Loaded Images into the tray.")
    print("Loaded Labels into the tray.")

    if channel_flag == 1:
        aux_db1 = list_dB[1]
        db_strain_img = root_db_path + aux_db1 + "/" + aux_db1 + "/"
        if cross_db_flag == 1:
            SubperdB = Read_Input_Images_SAMM_CASME(db_strain_img, list_samples, list_of_ignored_samples, aux_db1,
                                                    resizedFlag, table, db_home, spatial_size, 1)
        else:
            SubperdB_strain = Read_Input_Images(db_strain_img, list_of_ignored_samples, aux_db1, resizedFlag, table,
                                                db_home, spatial_size, 1, objective_flag)

    elif channel_flag == 2:
        aux_db1 = list_dB[1]
        aux_db2 = list_dB[2]
        db_strain_img = root_db_path + aux_db1 + "/" + aux_db1 + "/"
        db_gray_img = root_db_path + aux_db2 + "/" + aux_db2 + "/"
        if cross_db_flag == 1:
            SubperdB_strain = Read_Input_Images_SAMM_CASME(db_strain_img, list_samples, list_of_ignored_samples, aux_db1,
                                                           resizedFlag, table, db_home, spatial_size, 1)
            SubperdB_gray = Read_Input_Images_SAMM_CASME(db_gray_img, list_samples, list_of_ignored_samples, aux_db2,
                                                         resizedFlag, table, db_home, spatial_size, 1)
        else:
            SubperdB_strain = Read_Input_Images(db_strain_img, list_of_ignored_samples, aux_db1, resizedFlag, table,
                                                db_home, spatial_size, 1, objective_flag)
            SubperdB_gray = Read_Input_Images(db_gray_img, list_of_ignored_samples, aux_db2, resizedFlag, table, db_home,
                                              spatial_size, 1, objective_flag)

    elif channel_flag == 3:
        aux_db1 = list_dB[1]
        db_strain_img = root_db_path + aux_db1 + "/" + aux_db1 + "/"
        if cross_db_flag == 1:
            SubperdB = Read_Input_Images_SAMM_CASME(db_strain_img, list_samples, list_of_ignored_samples, aux_db1,
                                                    resizedFlag, table, db_home, spatial_size, 3)
        else:
            SubperdB_strain = Read_Input_Images(db_strain_img, list_of_ignored_samples, aux_db1, resizedFlag, table,
                                                db_home, spatial_size, 3, objective_flag)

    elif channel_flag == 4:
        aux_db1 = list_dB[1]
        aux_db2 = list_dB[2]
        db_strain_img = root_db_path + aux_db1 + "/" + aux_db1 + "/"
        db_gray_img = root_db_path + aux_db2 + "/" + aux_db2 + "/"
        if cross_db_flag == 1:
            SubperdB_strain = Read_Input_Images_SAMM_CASME(db_strain_img, list_samples, list_of_ignored_samples, aux_db1,
                                                           resizedFlag, table, db_home, spatial_size, 3)
            SubperdB_gray = Read_Input_Images_SAMM_CASME(db_gray_img, list_samples, list_of_ignored_samples, aux_db2,
                                                         resizedFlag, table, db_home, spatial_size, 3)
        else:
            SubperdB_strain = Read_Input_Images(db_strain_img, list_of_ignored_samples, aux_db1, resizedFlag, table,
                                                db_home, spatial_size, 3, objective_flag)
            SubperdB_gray = Read_Input_Images(db_gray_img, list_of_ignored_samples, aux_db2, resizedFlag, table, db_home,
                                              spatial_size, 3, objective_flag)

    #######################################################

    ########### Model Configurations #######################
    K.set_image_dim_ordering('th')
    sgd = optimizers.SGD(lr=0.0001, decay=1e-7, momentum=0.9, nesterov=True)
    adam = optimizers.Adam(lr=0.00001, decay=0.000001)

    # Different Conditions for Temporal Learning ONLY
    if train_spatial_flag == 0 and train_temporal_flag == 1 and dB != 'CASME2_Optical':
        data_dim = spatial_size * spatial_size
    elif train_spatial_flag == 0 and train_temporal_flag == 1 and dB == 'CASME2_Optical':
        data_dim = spatial_size * spatial_size * 3
    elif channel_flag == 3:
        data_dim = 8192
    elif channel_flag == 4:
        data_dim = 12288
    else:
        data_dim = 4096

    ########################################################

    print("Beginning training process.")
    ########### Training Process ############

    for sub in range(subjects):
        print(".starting subject" + str(sub))
        gpu_observer()
        spatial_weights_name = root_db_path + 'Weights/' + str(train_id) + '/vgg_spatial_' + str(train_id) + '_' + str(
            dB) + '_'
        spatial_weights_name_strain = root_db_path + 'Weights/' + str(train_id) + '/vgg_spatial_strain_' + str(
            train_id) + '_' + str(dB) + '_'
        spatial_weights_name_gray = root_db_path + 'Weights/' + str(train_id) + '/vgg_spatial_gray_' + str(
            train_id) + '_' + str(dB) + '_'

        temporal_weights_name = root_db_path + 'Weights/' + str(train_id) + '/temporal_ID_' + str(train_id) + '_' + str(
            dB) + '_'

        ae_weights_name = root_db_path + 'Weights/' + str(train_id) + '/autoencoder_' + str(train_id) + '_' + str(
            dB) + '_'
        ae_weights_name_strain = root_db_path + 'Weights/' + str(train_id) + '/autoencoder_strain_' + str(
            train_id) + '_' + str(dB) + '_'

        ############### Reinitialization & weights reset of models ########################

        temporal_model = temporal_module(data_dim=data_dim, timesteps_TIM=timesteps_TIM, classes=n_exp)
        temporal_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])

        if channel_flag == 1:
            vgg_model = VGG_16_4_channels(classes=n_exp, channels=4, spatial_size=spatial_size)

            if fine_tuning_flag == 1:
                for layer in vgg_model.layers[:33]:
                    layer.trainable = False

            vgg_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])

        elif channel_flag == 2:
            vgg_model = VGG_16_4_channels(classes=n_exp, channels=5, spatial_size=spatial_size)

            if fine_tuning_flag == 1:
                for layer in vgg_model.layers[:33]:
                    layer.trainable = False

            vgg_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])


        elif channel_flag == 3 or channel_flag == 4:
            vgg_model = VGG_16(spatial_size=spatial_size, classes=n_exp, channels=3, weights_path='VGG_Face_Deep_16.h5')
            vgg_model_strain = VGG_16(spatial_size=spatial_size, classes=n_exp, channels=3,
                                      weights_path='VGG_Face_Deep_16.h5')

            if fine_tuning_flag == 1:
                for layer in vgg_model.layers[:33]:
                    layer.trainable = False
                for layer in vgg_model_strain.layers[:33]:
                    layer.trainable = False

            vgg_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])
            vgg_model_strain.compile(loss='categorical_crossentropy', optimizer=adam,
                                     metrics=[metrics.categorical_accuracy])

            if channel_flag == 4:
                vgg_model_gray = VGG_16(spatial_size=spatial_size, classes=n_exp, channels=3,
                                        weights_path='VGG_Face_Deep_16.h5')

                if fine_tuning_flag == 1:
                    for layer in vgg_model_gray.layers[:33]:
                        layer.trainable = False

                vgg_model_gray.compile(loss='categorical_crossentropy', optimizer=adam,
                                       metrics=[metrics.categorical_accuracy])

        else:
            vgg_model = VGG_16(spatial_size=spatial_size, classes=n_exp, channels=3, weights_path='VGG_Face_Deep_16.h5')

            if fine_tuning_flag == 1:
                for layer in vgg_model.layers[:33]:
                    layer.trainable = False

            vgg_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy])

        # svm_classifier = SVC(kernel='linear', C=1)
        ####################################################################################

        ############ for tensorboard ###############
        if tensorboard_flag == 1:
            cat_path = tensorboard_path + str(sub) + "/"
            os.mkdir(cat_path)
            tbCallBack = tensorflow.keras.callbacks.TensorBoard(log_dir=cat_path, write_graph=True)

            cat_path2 = tensorboard_path + str(sub) + "spatial/"
            os.mkdir(cat_path2)
            tbCallBack2 = tensorflow.keras.callbacks.TensorBoard(log_dir=cat_path2, write_graph=True)
        #############################################

        Train_X, Train_Y, Test_X, Test_Y, Test_Y_gt, X, y, test_X, test_y = restructure_data(sub, SubperdB, label_per_sub,
                                                                                             subjects, n_exp, r, w,
                                                                                             timesteps_TIM, channel)

        # Special Loading for 4-Channel
        if channel_flag == 1:
            _, _, _, _, _, Train_X_Strain, Train_Y_Strain, Test_X_Strain, Test_Y_Strain = restructure_data(sub,
                                                                                                           SubperdB_strain,
                                                                                                           label_per_sub,
                                                                                                           subjects,
                                                                                                           n_exp, r, w,
                                                                                                           timesteps_TIM,
                                                                                                           1)

            # verify
            # sanity_check_image(Test_X_Strain, 1, spatial_size)

            # Concatenate Train X & Train_X_Strain
            X = np.concatenate((X, Train_X_Strain), axis=1)
            test_X = np.concatenate((test_X, Test_X_Strain), axis=1)

            total_channel = 4

        elif channel_flag == 2:
            _, _, _, _, _, Train_X_Strain, Train_Y_Strain, Test_X_Strain, Test_Y_Strain = restructure_data(sub,
                                                                                                           SubperdB_strain,
                                                                                                           label_per_sub,
                                                                                                           subjects,
                                                                                                           n_exp, r, w,
                                                                                                           timesteps_TIM,
                                                                                                           1)

            _, _, _, _, _, Train_X_Gray, Train_Y_Gray, Test_X_Gray, Test_Y_Gray = restructure_data(sub, SubperdB_gray,
                                                                                                   label_per_sub,
                                                                                                   subjects, n_exp, r,
                                                                                                   w, timesteps_TIM, 1)

            # Concatenate Train_X_Strain & Train_X & Train_X_gray
            X = np.concatenate((X, Train_X_Strain, Train_X_Gray), axis=1)
            test_X = np.concatenate((test_X, Test_X_Strain, Test_X_Gray), axis=1)

            total_channel = 5

        elif channel_flag == 3:
            _, _, _, _, _, Train_X_Strain, Train_Y_Strain, Test_X_Strain, Test_Y_Strain = restructure_data(sub,
                                                                                                           SubperdB_strain,
                                                                                                           label_per_sub,
                                                                                                           subjects,
                                                                                                           n_exp, r, w,
                                                                                                           timesteps_TIM,
                                                                                                           3)

        elif channel_flag == 4:
            _, _, _, _, _, Train_X_Strain, Train_Y_Strain, Test_X_Strain, Test_Y_Strain = restructure_data(sub,
                                                                                                           SubperdB_strain,
                                                                                                           label_per_sub,
                                                                                                           subjects,
                                                                                                           n_exp, r, w,
                                                                                                           timesteps_TIM,
                                                                                                           3)
            _, _, _, _, _, Train_X_Gray, Train_Y_Gray, Test_X_Gray, Test_Y_Gray = restructure_data(sub, SubperdB_gray,
                                                                                                   label_per_sub,
                                                                                                   subjects, n_exp, r,
                                                                                                   w, timesteps_TIM, 3)

        ############### check gpu resources ####################
        gpu_observer()
        ########################################################

        print("Beginning training & testing.")
        ##################### Training & Testing #########################

        if train_spatial_flag == 1 and train_temporal_flag == 1:

            print("Beginning spatial training.")
            # Spatial Training
            if tensorboard_flag == 1:
                vgg_model.fit(X, y, batch_size=batch_size, epochs=spatial_epochs, shuffle=True,
                              callbacks=[history, stopping, tbCallBack2])

            elif channel_flag == 3 or channel_flag == 4:
                vgg_model.fit(X, y, batch_size=batch_size, epochs=spatial_epochs, shuffle=True,
                              callbacks=[history, stopping])
                vgg_model_strain.fit(Train_X_Strain, y, batch_size=batch_size, epochs=spatial_epochs, shuffle=True,
                                     callbacks=[stopping])
                model_strain = record_weights(vgg_model_strain, spatial_weights_name_strain, sub, flag)
                output_strain = model_strain.predict(Train_X_Strain, batch_size=batch_size)
                if channel_flag == 4:
                    vgg_model_gray.fit(Train_X_Gray, y, batch_size=batch_size, epochs=spatial_epochs, shuffle=True,
                                       callbacks=[stopping])
                    model_gray = record_weights(vgg_model_gray, spatial_weights_name_gray, sub, flag)
                    output_gray = model_gray.predict(Train_X_Gray, batch_size=batch_size)

            else:
                vgg_model.fit(X, y, batch_size=batch_size, epochs=spatial_epochs, shuffle=True,
                              callbacks=[history, stopping])

            print(".record f1 and loss")
            # record f1 and loss
            record_loss_accuracy(db_home, train_id, dB, history)

            print(".save vgg weights")
            # save vgg weights
            model = record_weights(vgg_model, spatial_weights_name, sub, flag)

            print(".spatial encoding")
            # Spatial Encoding
            output = model.predict(X, batch_size=batch_size)

            # concatenate features for temporal enrichment
            if channel_flag == 3:
                output = np.concatenate((output, output_strain), axis=1)
            elif channel_flag == 4:
                output = np.concatenate((output, output_strain, output_gray), axis=1)

            features = output.reshape(int(Train_X.shape[0]), timesteps_TIM, output.shape[1])

            print("Beginning temporal training.")

            # Temporal Training
            if tensorboard_flag == 1:
                temporal_model.fit(features, Train_Y, batch_size=batch_size, epochs=temporal_epochs,
                                   callbacks=[tbCallBack])
            else:
                temporal_model.fit(features, Train_Y, batch_size=batch_size, epochs=temporal_epochs)

            print(".save temportal weights")
            # save temporal weights
            temporal_model = record_weights(temporal_model, temporal_weights_name, sub, 't')  # let the flag be t

            print("Beginning testing.")
            print(".predicting with spatial model")
            # Testing
            output = model.predict(test_X, batch_size=batch_size)
            if channel_flag == 3 or channel_flag == 4:
                output_strain = model_strain.predict(Test_X_Strain, batch_size=batch_size)
                if channel_flag == 4:
                    output_gray = model_gray.predict(Test_X_Gray, batch_size=batch_size)

            # concatenate features for temporal enrichment
            if channel_flag == 3:
                output = np.concatenate((output, output_strain), axis=1)
            elif channel_flag == 4:
                output = np.concatenate((output, output_strain, output_gray), axis=1)

            print(".outputing features")
            features = output.reshape(Test_X.shape[0], timesteps_TIM, output.shape[1])

            print(".predicting with temporal model")
            predict = temporal_model.predict_classes(features, batch_size=batch_size)
        ##############################################################

        #################### Confusion Matrix Construction #############
        print(predict)
        print(Test_Y_gt.astype(int))

        print(".writing predicts to file")
        file = open(db_home + 'Classification/' + 'Result/' + dB + '/predicts_' + str(train_id) + '.txt', 'a')
        file.write("predicts_sub_" + str(sub) + "," + (",".join(repr(e) for e in predict.astype(list))) + "\n")
        file.write(
            "actuals_sub_" + str(sub) + "," + (",".join(repr(e) for e in Test_Y_gt.astype(int).astype(list))) + "\n")
        file.close()

        ct = confusion_matrix(Test_Y_gt, predict)
        # check the order of the CT
        order = np.unique(np.concatenate((predict, Test_Y_gt)))

        # create an array to hold the CT for each CV
        mat = np.zeros((n_exp, n_exp))
        # put the order accordingly, in order to form the overall ConfusionMat
        for m in range(len(order)):
            for n in range(len(order)):
                mat[int(order[m]), int(order[n])] = ct[m, n]

        tot_mat = mat + tot_mat
        ################################################################

        #################### cumulative f1 plotting ######################
        microAcc = np.trace(tot_mat) / np.sum(tot_mat)
        [f1, precision, recall] = fpr(tot_mat, n_exp)

        file = open(db_home + 'Classification/' + 'Result/' + dB + '/f1_' + str(train_id) + '.txt', 'a')
        file.write(str(f1) + "\n")
        file.close()
        ##################################################################

        ################# write each CT of each CV into .txt file #####################
        record_scores(db_home, dB, ct, sub, order, tot_mat, n_exp, subjects)
        war = weighted_average_recall(tot_mat, n_exp, samples)
        uar = unweighted_average_recall(tot_mat, n_exp)
        print("war: " + str(war))
        print("uar: " + str(uar))
        ###############################################################################

        ################## free memory ####################

        del vgg_model
        del temporal_model
        del model
        del Train_X, Test_X, X, y

        if channel_flag == 1:
            del Train_X_Strain, Test_X_Strain, Train_Y_Strain, Train_Y_Strain
        elif channel_flag == 2:
            del Train_X_Strain, Test_X_Strain, Train_Y_Strain, Train_Y_Strain, Train_X_Gray, Test_X_Gray, Train_Y_Gray, Test_Y_Gray
        elif channel_flag == 3:
            del vgg_model_strain, model_strain
            del Train_X_Strain, Test_X_Strain, Train_Y_Strain, Train_Y_Strain
        elif channel_flag == 4:
            del Train_X_Strain, Test_X_Strain, Train_Y_Strain, Train_Y_Strain, Train_X_Gray, Test_X_Gray, Train_Y_Gray, Test_Y_Gray
            del vgg_model_gray, vgg_model_strain, model_gray, model_strain

        gc.collect()
    ###################################################
