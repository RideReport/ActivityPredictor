//
//  RandomForest.swift
//  Ride
//
//  Created by William Henderson on 3/7/16.
//  Copyright © 2016 Knock Softwae, Inc. All rights reserved.
//

import Foundation

class RandomForestManager {
    var _ptr: COpaquePointer!
    var classLables: [Int32]!
    var classCount = 0
    
    struct Static {
        static var onceToken : dispatch_once_t = 0
        static var sharedManager : RandomForestManager?
    }
    
    class var sharedForest:RandomForestManager {
        return Static.sharedManager!
    }
    
    class func startup() {
        if (Static.sharedManager == nil) {
            Static.sharedManager = RandomForestManager()
            dispatch_async(dispatch_get_main_queue()) {
                Static.sharedManager?.startup()
            }
        }
    }
    
    func startup() {
        let path = NSBundle(forClass: self.dynamicType).pathForResource("forest.cv", ofType: nil)
        let cpath = path?.cStringUsingEncoding(NSUTF8StringEncoding)
        
        _ptr = createRandomForestManager(Int32(MotionManager.sampleWindowSize), Int32(1/MotionManager.updateInterval), UnsafeMutablePointer(cpath!))
        self.classCount = Int(randomForestGetClassCount(_ptr))
        self.classLables = [Int32](count:self.classCount, repeatedValue:0)
        randomForestGetClassLabels(_ptr, UnsafeMutablePointer(self.classLables), Int32(self.classCount))
    }
    
    deinit {
        deleteRandomForestManager(_ptr)
    }

    private func accelerometerReadings(forSensorData sensorData:NSOrderedSet)->[AccelerometerReading] {
        var readings: [AccelerometerReading] = []
        
        for elem in sensorData {
            let data = elem as! SensorData
            let reading = AccelerometerReading(x: data.x.floatValue, y: data.y.floatValue, z: data.z.floatValue, t: Float(data.date.timeIntervalSinceReferenceDate))
            readings.append(reading)
        }
        
        return readings
    }
    
    func classify(sensorDataCollection: SensorDataCollection)
    {
        let accelVector = self.accelerometerReadings(forSensorData: sensorDataCollection.accelerometerAccelerations)
        let confidences = [Float](count:self.classCount, repeatedValue:0.0)
        
        randomForestClassifyAccelerometerSignal(_ptr, UnsafeMutablePointer(accelVector), Int32(accelVector.count), UnsafeMutablePointer(confidences), Int32(self.classCount))
        
        var classConfidences: [Int: Float] = [:]
    
        for (i, score) in confidences.enumerate() {
            classConfidences[Int(classLables[i])] = score
        }

        sensorDataCollection.setActivityTypePredictions(forClassConfidences: classConfidences)
        CoreDataManager.sharedManager.saveContext()
    }
}
