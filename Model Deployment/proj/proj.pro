QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    deepSORT_Headers/DeepAppearanceDescriptor/deepsort.cpp \
    deepSORT_Headers/DeepAppearanceDescriptor/model.cpp \
    deepSORT_Headers/KalmanFilter/kalmanfilter.cpp \
    deepSORT_Headers/KalmanFilter/linear_assignment.cpp \
    deepSORT_Headers/KalmanFilter/nn_matching.cpp \
    deepSORT_Headers/KalmanFilter/track.cpp \
    deepSORT_Headers/KalmanFilter/tracker.cpp \
    deepSORT_Headers/MunkresAssignment/hungarianoper.cpp \
    deepSORT_Headers/MunkresAssignment/munkres/munkres.cpp \
    main.cpp \
    mainwindow.cpp \
    ncnnmodelbase.cpp \
    yoloxinference.cpp

HEADERS += \
    deepSORT_Headers/DeepAppearanceDescriptor/dataType.h \
    deepSORT_Headers/DeepAppearanceDescriptor/deepsort.h \
    deepSORT_Headers/DeepAppearanceDescriptor/model.h \
    deepSORT_Headers/KalmanFilter/kalmanfilter.h \
    deepSORT_Headers/KalmanFilter/linear_assignment.h \
    deepSORT_Headers/KalmanFilter/nn_matching.h \
    deepSORT_Headers/KalmanFilter/track.h \
    deepSORT_Headers/KalmanFilter/tracker.h \
    deepSORT_Headers/MunkresAssignment/hungarianoper.h \
    deepSORT_Headers/MunkresAssignment/munkres/matrix.h \
    deepSORT_Headers/MunkresAssignment/munkres/munkres.h \
    mainwindow.h \
    ncnnmodelbase.h \
    objLibrary.h \
    yoloxinference.h \



FORMS += \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target



win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../tools/opencv_base/opencv-release/install/x64/vc16/lib/ -lopencv_world440
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../tools/opencv_base/opencv-release/install/x64/vc16/lib/ -lopencv_world440d

INCLUDEPATH += $$PWD/../../tools/opencv_base/opencv-release/install/include
DEPENDPATH += $$PWD/../../tools/opencv_base/opencv-release/install/include

#ncnn gpu
INCLUDEPATH += $$PWD/../../tools/ncnn/build-qt/install/include
                $$PWD/'../../../Program Files (x86)/vulkan/Include'

DEPENDPATH += $$PWD/../../tools/ncnn/build-qt/install/include

LIBS += -L$$PWD/../../tools/ncnn/build-qt/install/lib/ -lncnn \
        -L$$PWD/'../../../Program Files (x86)/vulkan/Lib/' -lvulkan-1 \
        -L$$PWD/../../tools/ncnn/build-qt/install/lib/ -lglslang \
        -L$$PWD/../../tools/ncnn/build-qt/install/lib/ -lOSDependent \
        -L$$PWD/../../tools/ncnn/build-qt/install/lib/ -lOGLCompiler \
        -L$$PWD/../../tools/ncnn/build-qt/install/lib/ -lSPIRV \

INCLUDEPATH+= D:\code_library\c_code\qt_code\proj\deepSORT_Headers\eigen

#ncnn cpu
#win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../tools/ncnn-master/build-vs2019/install/lib/ -lncnn
#else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../tools/ncnn-master/build-vs2019/install/lib/ -lncnnd
#else:unix: LIBS += -L$$PWD/../../../tools/ncnn-master/build-vs2019/install/lib/ -lncnn

#INCLUDEPATH += $$PWD/../../../tools/ncnn-master/build-vs2019/install/include
#DEPENDPATH += $$PWD/../../../tools/ncnn-master/build-vs2019/install/include

#egien

RESOURCES += \
    src.qrc

