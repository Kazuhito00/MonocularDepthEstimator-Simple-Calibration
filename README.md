# MonocularDepthEstimator-Simple-Calibration
https://user-images.githubusercontent.com/37477845/123132746-06d30480-d48a-11eb-9902-c27e52878bc5.mp4


単眼デプス推定で推定した相対距離をシンプルなキャリブレーションで絶対距離へ変換するプログラムです。<br>
2点以上の実測値から最小二乗法で1次関数へ近似します。

# Requirement 
* opencv-python 4.5.2.54 or later
* onnxruntime 1.5.2 or later

# Usage
以下コマンドで起動してください。<br>
画面上でマウス左クリックすることで実測値(cm)の入力用のポップアップが出ます。<br>
2点以上で実測値を入力するとマウスポインタ上の距離表示が推論値からキャリブレーション値に変わります。<br>
また、キーボードの「c」を押下することでキャリブレーションの指定をリセットすることが出来ます。
```bash
python main.py
```
実行時には、以下のオプションが指定可能です。
   
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：640
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：360

# Memo
利用するモデルを独自にカスタマイズする際には、main.pyの以下個所を変更してください。
```
def model_load():
    from midas_predictor.midas_predictor import MiDaSPredictor

    model_path = 'midas_predictor/midas_v2_1_small.onnx'
    model_type = 'small'

    midas_predictor = MiDaSPredictor(model_path, model_type)

    def model_run(image):
        result = midas_predictor(image)
        return result

    return model_run
```

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
MonocularDepthEstimator-Simple-Calibration is under [MIT License](LICENSE).
