# this is a readme file


## hw3

[作业说明链接](https://docs.google.com/presentation/d/1_6TJrFs3JGBsJpdRGLK1Fy_EiJlNvLm_lTZ9sjLsaKE/edit#slide=id.p1)

[代码说明链接](https://colab.research.google.com/drive/16a3G7Hh8Pv1X1PhZAUBEnZEkXThzDeHJ)

## hw4

[作业说明链接](https://docs.google.com/presentation/d/1W5-D0hqchrkVgQxwNLBDlydamCHx5yetzmwbUiksBAA/edit#slide=id.g7cd4f194f5_2_45)

[代码说明链接](https://colab.research.google.com/drive/16d1Xox0OW-VNuxDn1pvy2UXFIPfieCb9)

## hw5

[作业说明链接](https://docs.google.com/presentation/d/1VClvgyilAvohextY0tM3gD7YemXGSUrzLV0E8RjDnMU/edit#slide=id.p1)

[代码说明链接](https://colab.research.google.com/drive/1FbuTOevZTUO3IEVJLwSfwCdGnrBf3Qwv#scrollTo=7RPpq4tH7a5E)

## hw6

[作业说明链接](https://docs.google.com/presentation/d/1aQNgb0dA6aAplW3U8l1wxc6LDjo7gpEOyEL5zlLJwcg/edit#slide=id.p1)

[代码说明链接](https://colab.research.google.com/drive/1ePbuJwBwVsHkfztpXKjKuqaEZ3h27F_A)

## hw7

[作业说明链接](https://docs.google.com/presentation/d/1n5gc0uk3ysoOzfH2kd56DJwj-BE6le_CXiBboK9g8Hk/edit#slide=id.g7be340f71d_0_186)

[代码说明链接](https://colab.research.google.com/drive/1lJS0ApIyi7eZ2b3GMyGxjPShI8jXM2UC)

## hw8

[作业说明链接](https://docs.google.com/presentation/d/1xshFEjpgRgpB-lZNbdRV_BNP0rmh5sAnz4eZHgS5Cs0/edit)

[代码说明链接](https://colab.research.google.com/drive/11iwJbQv9iScRo6kGP7YfyHaaorlHhzMT)

## hw9

[作业说明链接](https://docs.google.com/presentation/d/1ULbTKqn7ikFOTU-r0DoqAca6lej3QmLWwORfcr-0F3o/edit#slide=id.g7be340f71d_0_186)

[代码说明链接](https://colab.research.google.com/drive/1sHOS6NFIBW5aZGz5RePyexFe28MvaPU6)

## hw10

[作业说明链接](https://docs.google.com/presentation/d/1kvYOBycYRs9P-nWrlZNnwmdnnAO6w69jKjCxmTRlNqU/edit#slide=id.p1)

[代码说明链接](https://colab.research.google.com/drive/12D52GgTwb4k75mRCSM_y8ykqHvqk_gOJ#scrollTo=cBo2oxu_WmZY)

1. 任取一個baseline model (sample code裡定義的 fcn，cnn，vae) 與你在kaggle leaderboard上表現最好的model（如果表現最好的model就是sample code裡定義的model的話就任選兩個，e.g.  fcn & cnn），對各自重建的testing data的image中選出與原圖mse最大的兩張加上最小的兩張並畫出來。（假設有五張圖，每張圖經由autoencoder A重建的圖片與原圖的MSE分別為 [25.4, 33.6, 15, 39, 54.8]，則MSE最大的兩張是圖4、5而最小的是圖1、3）。須同時附上原圖與經autoencoder重建的圖片。（圖片總數：(原圖+重建)*(兩顆model)*(mse最大兩張+mse最小兩張) = 16張）

2. 嘗試把 sample code中的KNN 與 PCA 做在 autoencoder 的 encoder output 上，並回報兩者的auc score。

3. 如hw9，使用PCA或T-sne將testing data投影在2維平面上，並將testing data經第1題的兩顆model的encoder降維後的output投影在2維平面上，觀察經encoder降維後是否分成兩群的情況更明顯。（因未給定testing label，所以點不須著色）（總共3張圖）

4. 說明為何使用auc score來衡量而非binary classification常用的f1 score。如果使用f1 score會有什麼不便之處？

