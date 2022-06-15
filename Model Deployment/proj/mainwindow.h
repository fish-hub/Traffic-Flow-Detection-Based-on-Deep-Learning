#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QMessageBox>
#include <QStringLiteral>
#include <QFileInfo>
#include <QFileDialog>
#include <QStatusBar>
#include <QSpinBox>
#include "opencv2/opencv.hpp"
#include "yoloxinference.h"
QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

    QLabel* imageLabelSrc;
    QLabel* imageLabelDst;
    QPushButton* btnLoadModelBin;
    QPushButton* btnLoadModelParam;
    QPushButton* btnLoadVideo;
    QPushButton* btnShowVideo;
    QPushButton* btnInference;
    QPushButton* btnSaveVideo;
    QSpinBox * yelloLineSpinBox;
    QSpinBox * blueLineSpinBox;
    QPushButton* exitSystem;
    QLabel* statusBarLabel;
    QLabel* lblcurmode;
    QString fileName = "D:\\code_library\\c_code\\proj\\video\\car1.mp4";
    int trafficFlowNum_Down=0;
    int trafficFlowNum_Up=0;
    bool exitFlag = 0;
    bool curmode=0;
    cv::Mat frame;
    int bias0 =0;
    int bias1=0;
    QFont infoFont;
    QFont statusBarFont;
    QFont toolBarFont;
    int screenWidth=0;
    int screenHeight=0;
    int labelWidth=0;
    int labelHeight=0;
    int labelX0=0;
    int labelY0=0;
    int labelX1=0;
    int labelY1=0;
    int btnSizeWidth=0;
    int btnSizeHeight=0;
    int btnX0=0;
    int btnY0=0;
    int btnStep=0;
    QString modelPathBin = "D:\\code_library\\c_code\\proj\\weights\\yolox_s-sim.bin";
    QString modelPathParam = "D:\\code_library\\c_code\\proj\\weights\\yolox_s-sim.param";
    QLabel* trafficFlowNums;
    void windowInit();
    QImage cvmat2QImage(cv::Mat src);
    int* calcuImageShape(int width, int height);
    void toolBarInit();
public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;

public slots:
    void inference();
    void spinBoxYelloBias_valueChanged(int);
    void spinBoxBlueBias_valueChanged(int);
};
#endif // MAINWINDOW_H
