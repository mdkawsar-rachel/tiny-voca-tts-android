plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "woof.tts"
    compileSdk = 34

    defaultConfig {
        applicationId = "woof.tts"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    androidResources {
        noCompress += listOf("onnx", "tsv")
    }
}

tasks.withType<Test>().configureEach {
    systemProperty("fixtures", rootProject.file("fixtures/all.tsv").absolutePath)
    systemProperty("respell", rootProject.file("app/src/main/assets/respell.tsv").absolutePath)
}

dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.3")
    testImplementation("junit:junit:4.13.2")
}