import org.gradle.api.JavaVersion

plugins {
    id("com.android.library") version "8.5.2"
    id("org.jetbrains.kotlin.android") version "1.9.25"
}

android {
    namespace = "com.securityresearch.capture"
    compileSdk = 34

    defaultConfig {
        minSdk = 26
        targetSdk = 34
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    testOptions {
        unitTests.isIncludeAndroidResources = true
    }
}

repositories {
    google()
    mavenCentral()
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.annotation:annotation:1.8.1")

    testImplementation("junit:junit:4.13.2")
    testImplementation("androidx.test:core:1.5.0")
    testImplementation(kotlin("test"))
}
