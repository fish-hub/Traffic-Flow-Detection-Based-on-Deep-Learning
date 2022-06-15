#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QDebug>
#include "./deepSORT_Headers/DeepAppearanceDescriptor/deepsort.h"
#include "./deepSORT_Headers/KalmanFilter/tracker.h"
#include "QTime"
#include "QFrame"
#include <QSpinBox>
#include <QScreen>
void postprocess(cv::Mat& frame, const  std::vector<Object>& out,   DETECTIONS& d);

void get_detections(DETECTBOX box,float confidence,DETECTIONS& d);

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->windowInit();
    this->toolBarInit();

    connect(this->btnLoadVideo,&QPushButton::clicked,this,[=](){
        QFileInfo fileInfo;
        QString fileNameFilter = QString("MP4 Files(*.mp4);;AVI Files(*.avi);;MKV Files(*.mkv)");
        QString tempfileName;
        tempfileName = QFileDialog::getOpenFileName(this,QStringLiteral("导入视频文件"),\
                                                QString("./videos"),fileNameFilter);
        fileInfo = QFileInfo(tempfileName);
        if(fileInfo.size()!=0)
        {
            this->fileName = tempfileName;
        }
    });

    connect(this->btnLoadModelBin,&QPushButton::clicked,this,[=](){
        QFileInfo fileInfo;
        QString fileNameFilter = QString("Model Bin(*.bin)");
        QString tempFileName;
        tempFileName = QFileDialog::getOpenFileName(this,QStringLiteral("导入模型bin"),\
                                                   QString("./weights"),fileNameFilter);
        fileInfo = QFileInfo(tempFileName);
        if(fileInfo.size()!=0)
        {
            QString statusInfo = QStringLiteral("导入模型bin成功，模型路径：%1").arg(tempFileName);
            this->statusBarLabel->setText(statusInfo);
            this->modelPathBin = tempFileName;
        }
    });

    connect(this->btnLoadModelParam,&QPushButton::clicked,this,[=](){
        QFileInfo fileInfo;
        QString fileNameFilter = QString("Model Param(*.param)");
        QString tempFileName;
        tempFileName = QFileDialog::getOpenFileName(this,QStringLiteral("导入模型param"),\
                                                   QString("./weights"),fileNameFilter);
        fileInfo = QFileInfo(tempFileName);
        if(fileInfo.size()!=0)
        {
            QString statusInfo = QStringLiteral("导入模型param成功，模型路径：%1").arg(tempFileName);
            this->statusBarLabel->setText(statusInfo);

            this->modelPathParam = tempFileName;
        }
    });

    connect(this->btnShowVideo,&QPushButton::clicked,this,[=](){
        cv::VideoCapture capture;
        capture.open(this->fileName.toStdString());
        cv::Mat frame;
        if(!capture.isOpened())
        {
            QMessageBox::critical(this,QStringLiteral("错误"),QStringLiteral("无法打开视频文件，请检查路径"));
            return -1;
        }
        while (this->exitFlag==0)
        {
            capture >> frame;
            int* afterShape = calcuImageShape(frame.cols,frame.rows);
            cv::resize(frame,frame,cv::Size(afterShape[0],afterShape[1]));
            QImage imageShow = this->cvmat2QImage(frame);
            this->imageLabelSrc->setPixmap(QPixmap::fromImage(imageShow));
            this->imageLabelSrc->setAlignment(Qt::AlignCenter);
            cv::waitKey(20);
        }

    });

    connect(this->btnInference,&QPushButton::clicked,this,&MainWindow::inference);

    connect(this->btnSaveVideo,&QPushButton::clicked,this,[=](){
        QDateTime curDateTime=QDateTime::currentDateTime();
        QString cur_time = curDateTime.time().toString("hh:mm:ss");
        QString cur_date = curDateTime.date().toString("yyyy-MM-dd");
        QFile myfile("out.txt");
        if (myfile.open(QFile::ReadWrite| QIODevice::Append))//|QFile::Truncate))//注意WriteOnly是往文本中写入的时候用，ReadOnly是在读文本中内容的时候用，Truncate表示将原来文件中的内容清空
        {
            QTextStream out(&myfile);
            //out<<myfile.readAll();
            out<<"\n";
            out<<"cur_date:"<<cur_date<<"\n";
            out<<"cur_time:"<<cur_time<<"\n";
            out<<"trafficFlowNum_Up:"<<trafficFlowNum_Up<<"\n";
            out<<"trafficFlowNum_Down:"<<trafficFlowNum_Down<<"\n";
            out<<"trafficFlowNum_Sum:"<<trafficFlowNum_Up+trafficFlowNum_Down<<endl;
            QMessageBox::information(this,QStringLiteral("提示"),QStringLiteral("保存成功"));
        }
    });


    connect(this->exitSystem,&QPushButton::clicked,this,[=](){
        if (!(QMessageBox::information(this,QStringLiteral("退出"),QStringLiteral("确认退出系统？"),QStringLiteral("是"),QStringLiteral("否"))))
             {
                   this->exitFlag=1;
                   QApplication* app;
                   app->exit(0);
             }

    });

    connect(ui->actionmode,&QAction::triggered,this,[=](){
        if(this->curmode==0)
        {
            this->curmode=1;
            QMessageBox::information(this,QStringLiteral("提示"),QStringLiteral("切换成功，当前为车辆检测模式"));
            this->lblcurmode->setText(QStringLiteral("检测"));
        }
        else
        {
            this->curmode=0;
            QMessageBox::information(this,QStringLiteral("提示"),QStringLiteral("切换成功，当前为车流量统计模式"));
            this->lblcurmode->setText(QStringLiteral("计数"));
        }
    });


    void (QSpinBox:: *valueChanged)(int) = &QSpinBox::valueChanged;
    connect(this->yelloLineSpinBox,valueChanged,this,&MainWindow::spinBoxYelloBias_valueChanged);
    connect(this->blueLineSpinBox,valueChanged,this,&MainWindow::spinBoxBlueBias_valueChanged);
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::inference()
{
    if(this->fileName=="")QMessageBox::critical(this,QStringLiteral("错误"),QStringLiteral("无视频文件"));
    else if(this->modelPathBin=="")QMessageBox::critical(this,QStringLiteral("错误"),QStringLiteral("无模型bin文件"));
    else if(this->modelPathParam=="")QMessageBox::critical(this,QStringLiteral("错误"),QStringLiteral("无模型param文件"));
    else
    {
        yoloxInference yolox;
        cv::VideoCapture capture;
        capture.open(this->fileName.toStdString());

       // capture.set(cv::CAP_PROP_POS_FRAMES,1000);
        capture >> frame;
        cv::Mat frameLeft;
        cv::Mat frameRight;
        int* afterShape = calcuImageShape(frame.cols,frame.rows);
        int resolution = frame.cols * frame.rows;
        yolox.loadModel(this->modelPathBin.toStdString(),this->modelPathParam.toStdString(),frame);
        if(this->curmode==0)
        {
            this->yelloLineSpinBox->setMaximum(frame.rows);
            this->blueLineSpinBox->setMaximum(frame.rows);
            this->yelloLineSpinBox->setValue(int(frame.rows*3/5+frame.rows/150.));
            this->blueLineSpinBox->setValue(int(frame.rows*4/5));

            //deepsort param
            const int nn_budget = 100;
            const float max_cosine_distance = 0.2;
            tracker mytracker(max_cosine_distance,nn_budget);
            DeepSort deepsort;

            std::set<std::string>trafficFlow_in;
            std::set<std::string>trafficFlow_out;
            float fontScale = resolution / (2000 * 1000);
            if(fontScale<0.5)fontScale=1;
            else if(fontScale<1)fontScale=1.5;
            while(this->exitFlag==0)
            {
                double time = (double)cv::getTickCount();
                capture >> frame;
                frameRight = frame.clone();
                cv::Mat mask = cv::Mat::zeros(frameRight.rows,frameRight.cols,CV_8UC3);

                cv::Rect rectYello(0,bias0,frame.cols,frame.rows/20.);
                cv::Rect rectBlue(0,bias1,frame.cols,frame.rows/20.);

                rectangle(mask,rectYello,cv::Scalar(0,255,255),-1);
                rectangle(mask,rectBlue,cv::Scalar(255,0,0),-1);
                cv::addWeighted(frameRight,1,mask,0.5,0,frameRight);


                std::vector<Object>* pp = yolox.startInference(frameRight);
                std::vector<Object>* proposals = new std::vector<Object> ;

                for(int i=0;i<pp->size();i++)
                {
                    if((*pp)[i].rect.width==0 ||(*pp)[i].rect.height==0)
                        continue;
                    proposals->push_back((*pp)[i]);
                }

                DETECTIONS detections;
                postprocess(frame,(*proposals),detections);
                if(deepsort.getRectsFeature(frame,detections))
                {
                    mytracker.predict();
                    mytracker.update(detections);
                    std::vector<RESULT_DATA> result;
                    for(Track& track : mytracker.tracks) {
                        if(!track.is_confirmed() || track.time_since_update > 1) continue;
                        result.push_back(std::make_pair(track.track_id, track.to_tlwh()));
                    }

                    for(unsigned int k = 0; k < detections.size(); k++)
                    {
                        DETECTBOX tmpbox = detections[k].tlwh;
                        cv::Rect rect(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
                        //cv::rectangle(frameRight, rect, cv::Scalar(0,0,255), 4);

                        for(unsigned int k = 0; k < result.size(); k++)
                        {
                            DETECTBOX tmp = result[k].second;
                            cv::Rect rect = cv::Rect(tmp(0), tmp(1), tmp(2), tmp(3));
                            rectangle(frameRight, rect, cv::Scalar(255, 255, 0), 2);

                            std::string label = cv::format("%d", result[k].first);

                            cv::putText(frameRight, label, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 0), 2);
                            cv::Point centerPoint = cv::Point(int(tmp(0)+tmp(2)/2.),int(tmp(1)+tmp(3)/2.));
                            cv::circle(frameRight,centerPoint,2,cv::Scalar(0,0,255),4);
                            //点进入黄色区域
                            if(rectYello.y<centerPoint.y && centerPoint.y<(rectYello.height+rectYello.y))
                            {

                                cv::circle(frameRight,centerPoint,5,cv::Scalar(238,238,0),8);
                                trafficFlow_in.insert(label);
                                for(set<std::string>::const_iterator iter=trafficFlow_out.begin();iter!=trafficFlow_out.end();++iter)
                                {
                                    if(*iter ==label)
                                    {
                                       trafficFlowNum_Up+=1;
                                       trafficFlow_out.erase(iter);

                                    }
                                }
                            }
                            //点进入蓝色区域
                            if(rectBlue.y<centerPoint.y && centerPoint.y<(rectBlue.height+rectBlue.y))
                            {
                                cv::circle(frameRight,centerPoint,5,cv::Scalar(0,255,0),8);
                                trafficFlow_out.insert(label);
                                for(set<std::string>::const_iterator iter=trafficFlow_in.begin();iter!=trafficFlow_in.end();++iter)
                                {
                                    if(*iter ==label)
                                    {
                                       trafficFlowNum_Down+=1;
                                       trafficFlow_in.erase(iter);

                                    }
                                }
                            }
                        }


                    }
        }

                cv::putText(frameRight,QString("Number of traffic flow down:%1").arg(trafficFlowNum_Down).toStdString(),
                            cv::Point(20,20*fontScale),cv::FONT_HERSHEY_DUPLEX,fontScale,cv::Scalar(0,0,255),fontScale+1);
                cv::putText(frameRight,QString("Number of traffic flow up:%1").arg(trafficFlowNum_Up).toStdString(),
                            cv::Point(20,40*fontScale),cv::FONT_HERSHEY_DUPLEX,fontScale,cv::Scalar(0,0,255),fontScale+1);
                cv::putText(frameRight,QString("Number of targets detected:%1").arg(detections.size()).toStdString(),
                            cv::Point(20,60*fontScale),cv::FONT_HERSHEY_DUPLEX,fontScale,cv::Scalar(0,0,255),fontScale+1);

                this->trafficFlowNums->setText(QString("%1").arg(trafficFlowNum_Down+trafficFlowNum_Up));

                time = ((double)cv::getTickCount() - time) / cv::getTickFrequency();
                cv::Mat imgDst;
                cv::resize(frame,frameLeft,cv::Size(afterShape[0],afterShape[1]));
                cv::resize(frameRight,imgDst,cv::Size(afterShape[0],afterShape[1]));

                QImage imageLeft = this->cvmat2QImage(frameLeft);
                QImage imageRight = this->cvmat2QImage(imgDst);
                this->imageLabelSrc->setPixmap(QPixmap::fromImage(imageLeft));
                this->imageLabelDst->setPixmap(QPixmap::fromImage(imageRight));
                cv::waitKey(10);
        }
    }
    else
        {
            float fontScale = resolution / (2000 * 1000);
            if(fontScale<0.5)fontScale=1.5;
            else if(fontScale<1)fontScale=2;

            float picInfoFontScale = resolution / (2000 * 2000);
            if(picInfoFontScale<0.5)picInfoFontScale=1;
            else if(picInfoFontScale<1)picInfoFontScale=1.5;
            while(this->exitFlag==0)
            {
                capture >> frame;
                frameRight = frame.clone();
                std::vector<Object>* proposals = yolox.startInference(frameRight);

//                cv::Mat gg = yolox.draw_objects(frame,*proposals);
//                cv::imshow("dd",gg);
                DETECTIONS detections;
                postprocess(frame,(*proposals),detections);
                for (int k=0; k<detections.size();k++)
                {
                    DETECTBOX tmpbox = detections[k].tlwh;
                    cv::Rect rect(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
                    cv::rectangle(frameRight, rect, cv::Scalar(0,0,255), 4);
                    char text[256];
                    Object obj = (*proposals)[k];

                    sprintf(text, "car %.1f%%", obj.prob * 100);

                    cv::putText(frameRight, text, cv::Point(rect.x, rect.y),
                                cv::FONT_HERSHEY_SIMPLEX, picInfoFontScale,
                                cv::Scalar(0,0,255), picInfoFontScale*2);
                }

                cv::putText(frameRight,QString("Number of targets detected:%1").arg((*proposals).size()).toStdString(),
                            cv::Point(20,60*fontScale+1),cv::FONT_HERSHEY_DUPLEX,fontScale,cv::Scalar(0,255,255),fontScale+3);

                cv::Mat imgDst;
                cv::resize(frame,frameLeft,cv::Size(afterShape[0],afterShape[1]));
                cv::resize(frameRight,imgDst,cv::Size(afterShape[0],afterShape[1]));

                QImage imageLeft = this->cvmat2QImage(frameLeft);
                QImage imageRight = this->cvmat2QImage(imgDst);
                this->imageLabelSrc->setPixmap(QPixmap::fromImage(imageLeft));
                this->imageLabelDst->setPixmap(QPixmap::fromImage(imageRight));

                cv::waitKey(50);
                }
        }

    }
}

void MainWindow::spinBoxYelloBias_valueChanged(int value)
{
    this->bias0 = value;

}

void MainWindow::spinBoxBlueBias_valueChanged(int value)
{
    this->bias1 = value;
}
void MainWindow::windowInit()
{
    //this->setFixedSize(2500,1200);
    this->setWindowState(Qt::WindowMaximized);

    QScreen *screenPrimary = QGuiApplication::primaryScreen();
    QRect screenSize = screenPrimary->availableGeometry();
    this->screenWidth = screenSize.width();
    this->screenHeight = screenSize.height();
    this->labelWidth = int(this->screenWidth/2.2);
    this->labelHeight = int(this->screenHeight/1.25);

    //qDebug() << screenSize.width() << screenSize.height();
    this->setWindowTitle(QStringLiteral("车流量检测平台"));
    this->setWindowIcon(QIcon(":/imgs/src/2.png"));
    QString btnStyleSheet = "QPushButton{color:black;background:white}"
                         "QPushButton::hover{background:rgb(110,115,100);}"
                         "QPushButton::pressed{background:#eb7350;}"
                         "QPushButton{font-family:SimSun;font: 25px}"
                         "QPushButton{border-radius:20;border:3px solid black}"
                         ;
    QString exitStyleSheet = "QPushButton{color:black;background:rgb(200,100,0)}"
                         "QPushButton::hover{background:rgb(110,115,100);}"
                         "QPushButton::pressed{background:#eb7350;}"
                         "QPushButton{font-family:SimSun;font: 25px}"
                         "QPushButton{border-radius:20;border:3px solid black}"
                         ;


    this->imageLabelSrc = new QLabel(this);
    this->imageLabelDst = new QLabel(this);
    this->btnLoadModelBin = new QPushButton(this);
    this->btnLoadModelParam = new QPushButton(this);
    this->btnInference = new QPushButton(this);
    this->btnLoadVideo = new QPushButton(this);
    this->btnSaveVideo = new QPushButton(this);
    this->btnShowVideo = new QPushButton(this);
    this->statusBarLabel = new QLabel(this);
    this->trafficFlowNums = new QLabel(this);
    this->lblcurmode = new QLabel(this);
    this->exitSystem = new QPushButton(this);

    this->labelX0 = 10;
    this->labelX1 = this->screenWidth-10-this->labelWidth;
    this->labelY0 = this->labelY1 = 100;
    this->btnSizeWidth = int(3./4.*(this->labelX1-this->labelX0-this->labelWidth));
    this->btnSizeHeight = 50;
    this->btnStep = int((this->labelHeight*3./4.)/8.);
    this->btnX0 = (this->labelX1-(this->labelX0+this->labelWidth))/2- this->btnSizeWidth/2+this->labelX0+this->labelWidth;
    this->btnY0 = int(this->labelY0 + this->labelHeight*1./8);

    //  左label
    this->imageLabelSrc->resize(this->labelWidth,this->labelHeight);
    this->imageLabelSrc->setFrameShape(QFrame::Panel);
    this->imageLabelSrc->move(this->labelX0,this->labelY0);

    //右label
    this->imageLabelDst->resize(this->labelWidth,this->labelHeight);
    this->imageLabelDst->setFrameShape(QFrame::Panel);
    this->imageLabelDst->move(this->labelX1,this->labelY1);

    this->btnLoadModelBin->resize(this->btnSizeWidth,this->btnSizeHeight);
    this->btnLoadModelParam->resize(this->btnSizeWidth,this->btnSizeHeight);
    this->btnLoadVideo->resize(this->btnSizeWidth,this->btnSizeHeight);
    this->btnSaveVideo->resize(this->btnSizeWidth,this->btnSizeHeight);
    this->btnInference->resize(this->btnSizeWidth,this->btnSizeHeight);
    this->btnShowVideo->resize(this->btnSizeWidth,this->btnSizeHeight);
    this->exitSystem->resize(this->btnSizeWidth,this->btnSizeHeight);

    this->btnLoadModelBin->move(this->btnX0,this->btnY0+this->btnStep*1);
    this->btnLoadModelParam->move(this->btnX0,this->btnY0+this->btnStep*2);
    this->btnLoadVideo->move(this->btnX0,this->btnY0+this->btnStep*3);
    this->btnShowVideo->move(this->btnX0,this->btnY0+this->btnStep*4);
    this->btnInference->move(this->btnX0,this->btnY0+this->btnStep*5);
    this->btnSaveVideo->move(this->btnX0,this->btnY0+this->btnStep*6);
    this->exitSystem->move(this->btnX0,this->btnY0+this->btnStep*7);

    this->btnLoadModelBin->setText(QStringLiteral("导入bin"));
    this->btnLoadModelBin->setStyleSheet(btnStyleSheet);

    this->btnLoadModelParam->setText(QStringLiteral("导入param"));
    this->btnLoadModelParam->setStyleSheet(btnStyleSheet);

    this->btnLoadVideo->setText(QStringLiteral("导入视频"));
    this->btnLoadVideo->setStyleSheet(btnStyleSheet);

    this->btnSaveVideo->setText(QStringLiteral("保存结果"));
    this->btnSaveVideo->setStyleSheet(btnStyleSheet);

    this->btnShowVideo->setText(QStringLiteral("播放视频"));
    this->btnShowVideo->setStyleSheet(btnStyleSheet);

    this->btnInference->setText(QStringLiteral("模型推理"));
    this->btnInference->setStyleSheet(btnStyleSheet);

    this->exitSystem->setText(QStringLiteral("退出系统"));
    this->exitSystem->setStyleSheet(exitStyleSheet);

    //状态栏字体调整
    this->infoFont.setFamily("SimSun");
    this->infoFont.setPointSize(20);
    this->infoFont.setBold(1);

    this->statusBarFont.setFamily("SimSun");
    this->statusBarFont.setPointSize(17);
    this->statusBarFont.setBold(1);

    //状态栏显示当前模式
    QLabel* curmodeInfo = new QLabel(QStringLiteral("当前模式："));
    curmodeInfo->setFont(this->infoFont);
    this->statusBar()->addWidget(curmodeInfo);
    this->lblcurmode->setText(QStringLiteral("计数"));
    this->lblcurmode->setFont(statusBarFont);
    this->lblcurmode->setStyleSheet("color:red;");
    this->statusBar()->addWidget(this->lblcurmode);

    //状态栏显示当前车辆数
    QLabel* trafficFlowInfo = new QLabel(QStringLiteral(" 车流量检测数目："));
    trafficFlowInfo->setFont(this->infoFont);
    this->trafficFlowNums->setText(QString("%1").arg(0));
    this->trafficFlowNums->setFont(statusBarFont);
    this->trafficFlowNums->setStyleSheet("color:red;");

    this->statusBar()->addWidget(trafficFlowInfo);
    this->statusBar()->addWidget(this->trafficFlowNums);

    // 设置背景
    this->setAutoFillBackground(true);
    QPalette p = this->palette();
    QPixmap pix(":/imgs/1.jpg");
    p.setBrush(QPalette::Window, QBrush(pix));
    this->setPalette(p);
}

QImage MainWindow::cvmat2QImage(cv::Mat src)
{
    QImage tempImage;
    cv::cvtColor(src,src,cv::COLOR_BGR2RGB);
    tempImage = QImage((const uchar*)(src.data), src.cols, src.rows,
                            src.cols * src.channels(), QImage::Format_RGB888);
    return tempImage;
}

int *MainWindow::calcuImageShape(int width, int height)
{
    float ratio = 1;
    int afterShape[2] = {width, height};
    if(width >= height)
    {
        ratio = float(width)/float(this->screenHeight);
        afterShape[0] = this->screenHeight;
        afterShape[1] = int(height/ratio);
    }
    else
    {
        ratio = float(height)/float(this->screenWidth);
        afterShape[1] = this->screenWidth;
        afterShape[0] = int(width/ratio);
    }

    return afterShape;
}

void MainWindow::toolBarInit()
{
    ui->toolBar->setIconSize(QSize(50,50));
    QLabel *lblLine1 = new QLabel;
    QLabel *lblLine2 = new QLabel;
    this->yelloLineSpinBox = new QSpinBox;
    this->blueLineSpinBox = new QSpinBox;

    // 工具栏字体
    this->toolBarFont.setFamily("SimSun");
    this->toolBarFont.setPointSize(10);
    this->toolBarFont.setBold(0);

    lblLine1->setText(QStringLiteral("检测线1位置: "));
    lblLine2->setText(QStringLiteral("检测线2位置: "));
    lblLine1->setFont(this->toolBarFont);
    lblLine2->setFont(this->toolBarFont);

    ui->actionmode->setText(QStringLiteral("模式切换"));
    ui->actionmode->setFont(toolBarFont);

    ui->toolBar->addWidget(lblLine1);
    ui->toolBar->addWidget(yelloLineSpinBox);
    this->yelloLineSpinBox->setMaximum(0);
    this->yelloLineSpinBox->setMinimum(0);
    this->yelloLineSpinBox->setSuffix(QString(" pixel"));
    this->yelloLineSpinBox->setSingleStep(50);
    this->yelloLineSpinBox->setValue(0);

    ui->toolBar->addWidget(lblLine2);
    ui->toolBar->addWidget(blueLineSpinBox);
    this->blueLineSpinBox->setMaximum(0);
    this->blueLineSpinBox->setMinimum(0);
    this->blueLineSpinBox->setValue(0);
    this->blueLineSpinBox->setSuffix(QString(" pixel"));
    this->blueLineSpinBox->setSingleStep(50);
}

void postprocess(cv::Mat& frame, const  std::vector<Object>& outs,DETECTIONS& d)
{
    for(const Object &info: outs)
    {
        get_detections(DETECTBOX(info.rect.x, info.rect.y,info.rect.width,  info.rect.height),
                       info.prob,d);
    }
}

void get_detections(DETECTBOX box,float confidence,DETECTIONS& d)
{
    DETECTION_ROW tmpRow;
    tmpRow.tlwh = box;//DETECTBOX(x, y, w, h);

    tmpRow.confidence = confidence;
    d.push_back(tmpRow);
}
