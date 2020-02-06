package com.nhean.faceverification

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import kotlinx.android.synthetic.main.activity_main.*
import androidx.core.app.ComponentActivity.ExtraData
import androidx.core.content.ContextCompat.getSystemService
import android.icu.lang.UCharacter.GraphemeClusterBreak.T
import android.net.Uri
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.core.app.ActivityCompat
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.io.OutputStream
import kotlin.system.exitProcess


class MainActivity : AppCompatActivity() {

    var image_path = ArrayList<String>()
    var image_path_str = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        ActivityCompat.requestPermissions(
            this@MainActivity,
            arrayOf(
                Manifest.permission.CAMERA,
                Manifest.permission.READ_EXTERNAL_STORAGE,
                Manifest.permission.WRITE_EXTERNAL_STORAGE
            ),
            PERMISSION_REQUEST
        )

        btn_select_image.setOnClickListener() {
            val intent = Intent()
            intent.type = "image/*"
            intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true)
            intent.action = Intent.ACTION_GET_CONTENT
            startActivityForResult(Intent.createChooser(intent, "Select Picture"), IMAGE_SELECTED_CODE)
        }

        btn_get128d.isEnabled = false
        btn_get128d.isClickable = false

        btn_get128d.setOnClickListener() {
            it.isEnabled = false
            it.isClickable = false

            Log.d("$TAG-Button", "btn_get128d is working")
            copyDlibModelFile("shape_predictor_5_face_landmarks.dat")
            copyDlibModelFile("dlib_face_recognition_resnet_model_v1.dat")

            Log.d("$TAG-Button", "Files are copied")
            var distance = get128DFromMat(image_path_str)
            txt_size.text = "Distance: $distance"

            Log.d("$TAG-Button", "Finished Task")

            it.isEnabled = true
            it.isClickable = true
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int,permissions: Array<String>,grantResults: IntArray) {
        when (requestCode) {
            PERMISSION_REQUEST -> {
                if (grantResults.isEmpty() && grantResults[0] != PackageManager.PERMISSION_GRANTED) {
                    moveTaskToBack(true)
                    exitProcess(-1)
                }
            }
            else -> {
                Log.e("$TAG-Permission", "Unexpected permission request")
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == Activity.RESULT_OK && requestCode == IMAGE_SELECTED_CODE){
            val clipData = data!!.getClipData()
            if (clipData != null)
            {
                for (i in 0 until clipData.getItemCount()){
                    Log.d("$TAG-PickImage", clipData.getItemAt(i).uri.toString())
                    image_path.add(convertUriToMat(clipData.getItemAt(i).uri).nativeObjAddr.toString())
                }
                image_path_str = image_path.toString().replace("[", "").replace("]", "").replace(" ","")
                btn_get128d.isEnabled = true
                btn_get128d.isClickable = true
            }
        }
    }

    private fun convertUriToMat(imageUri: Uri?) : Mat {
        var bmpFactoryOptions = BitmapFactory.Options()
        bmpFactoryOptions.inPreferredConfig = Bitmap.Config.ARGB_8888;
        var bmp = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
        var imageMat = Mat(bmp.width, bmp.height, CvType.CV_8UC4)
        Utils.bitmapToMat(bmp, imageMat)
        return imageMat
    }

    private fun copyDlibModelFile(filename:String) {
        val baseDir = Environment.getExternalStorageDirectory().getPath()
        val pathDir = baseDir + File.separator + filename
        val assetManager = this.getAssets()
        var inputStream: InputStream?
        var outputStream: OutputStream?

        Log.d("$TAG-CopyModel", pathDir)

        var full_path = File(pathDir)
        if(!full_path.exists()){
            Log.d("$TAG-CopyModel", "Model File is copying")
            try
            {
                inputStream = assetManager.open(filename)
                outputStream = FileOutputStream(pathDir)
                val buffer = ByteArray(1024)
                inputStream.use { input ->
                    outputStream.use { fileOut ->
                        while (true) {
                            val read = input.read(buffer)
                            if (read <= 0)
                                break
                            fileOut.write(buffer, 0, read)
                        }
                        fileOut.flush()
                        fileOut.close()
                    }
                }
                inputStream.close()
            }
            catch (e:Exception) {
                Log.e("$TAG-CopyModel", e.toString())
            }
        }
        else{
            Log.d("$TAG-CopyModel", "Model File is already copied!")
        }
    }

    // Load Native Library
    external fun get128DFromMat(image_addr: String): String

    companion object {
        const val PERMISSION_REQUEST = 101
        const val IMAGE_SELECTED_CODE = 102
        const val TAG = "MainActivity"
        init {
            System.loadLibrary("native-lib")
        }
    }
}
