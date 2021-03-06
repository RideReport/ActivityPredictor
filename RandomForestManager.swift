//
//  RandomForest.swift
//  Ride
//
//  Created by William Henderson on 3/7/16.
//  Copyright © 2016 Knock Softwae, Inc. All rights reserved.
//

import Foundation

class RandomForestManager {
    private var modelIdentifier: String?
    
    var _ptr: OpaquePointer!
    var classLables: [Int32]!
    var classCount = 0
    var desiredSampleInterval: TimeInterval {
        get {
            return Double(randomForestGetDesiredSamplingInterval(_ptr))
        }
    }
    
    var desiredSessionDuration: TimeInterval {
        get {
            return Double(randomForestGetDesiredSessionDuration(_ptr))
        }
    }
    
    var canPredict: Bool {
        return randomForestManagerCanPredict(_ptr)
    }
    
    init () {
        guard let configFilePath = Bundle(for: type(of: self)).path(forResource: "config.json", ofType: nil) else {
            return
        }
        
        let cConfigFilepath = configFilePath.cString(using: String.Encoding.utf8)
        
        _ptr = createRandomForestManagerFromFile(UnsafeMutablePointer(mutating: cConfigFilepath!))
    }
    
    deinit {
        deleteRandomForestManager(_ptr)
    }
    
    func startup() {
        guard let modelUIDCString = randomForestGetModelUniqueIdentifier(_ptr) else {
            return
        }

        modelIdentifier = String(cString: modelUIDCString)
        guard let modelID = modelIdentifier else {
            return
        }
        
        guard modelID.characters.count > 0  else {
            return
        }
        
        guard let modelPath = Bundle(for: type(of: self)).path(forResource: String(format: "%@.cv", modelID), ofType: nil) else {
            return
        }
        
        let cModelpath = modelPath.cString(using: String.Encoding.utf8)
        randomForestLoadModel(_ptr, UnsafeMutablePointer(mutating: cModelpath!))
        
        self.classCount = Int(randomForestGetClassCount(_ptr))
        self.classLables = [Int32](repeating: 0, count: self.classCount)
        randomForestGetClassLabels(_ptr, UnsafeMutablePointer(mutating: self.classLables), Int32(self.classCount))
    }

    private func accelerometerReadings(forSensorData sensorData:NSOrderedSet)->[AccelerometerReading] {
        var readings: [AccelerometerReading] = []
        
        for elem in sensorData {
            let data = elem as! SensorData
            let reading = AccelerometerReading(x: data.x.floatValue, y: data.y.floatValue, z: data.z.floatValue, t: data.date.timeIntervalSinceReferenceDate)
            readings.append(reading)
        }
        
        return readings
    }
    
    func classify(_ sensorDataCollection: SensorDataCollection)
    {
        let accelVector = self.accelerometerReadings(forSensorData: sensorDataCollection.accelerometerAccelerations)
        let confidences = [Float](repeating: 0.0, count: self.classCount)
        
        randomForestClassifyAccelerometerSignal(_ptr, UnsafeMutablePointer(mutating: accelVector), Int32(accelVector.count), UnsafeMutablePointer(mutating: confidences), Int32(self.classCount))
        
        var classConfidences: [Int: Float] = [:]
    
        for (i, score) in confidences.enumerated() {
            classConfidences[Int(classLables[i])] = score
        }

        sensorDataCollection.activityPredictionModelIdentifier = modelIdentifier
        sensorDataCollection.setActivityTypePredictions(forClassConfidences: classConfidences)
        CoreDataManager.shared.saveContext()
    }
}
