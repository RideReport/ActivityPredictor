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
        
        _ptr = createRandomForestManager(Int32(MotionManager.sampleWindowSize), Int32(1/MotionManager.updateInterval), UnsafeMutablePointer(cpath!), false)
        self.classCount = Int(randomForestGetClassCount(_ptr))
        self.classLables = [Int32](count:self.classCount, repeatedValue:0)
        randomForestGetClassLabels(_ptr, UnsafeMutablePointer(self.classLables), Int32(self.classCount))
    }
    
    deinit {
        deleteRandomForestManager(_ptr)
    }

    private func magnitudeVector(forSensorData sensorData:NSOrderedSet)->[Float] {
        var mags: [Float] = []
        
        for elem in sensorData {
            let reading = elem as! SensorData
            let sum = reading.x.floatValue*reading.x.floatValue + reading.y.floatValue*reading.y.floatValue + reading.z.floatValue*reading.z.floatValue
            mags.append(sqrtf(sum))
            if mags.count >= MotionManager.sampleWindowSize { break } // it is possible we over-colleted some of the sensor data
        }
        
        return mags
    }
    
    func classify(sensorDataCollection: SensorDataCollection)
    {
        let accelVector = self.magnitudeVector(forSensorData: sensorDataCollection.accelerometerAccelerations)
        let gyroVector = self.magnitudeVector(forSensorData: sensorDataCollection.gyroscopeRotationRates)

        let confidences = [Float](count:self.classCount, repeatedValue:0.0)
        randomForestClassificationConfidences(_ptr, UnsafeMutablePointer(accelVector), UnsafeMutablePointer(gyroVector), UnsafeMutablePointer(confidences), Int32(self.classCount))
        
        var classConfidences: [Int: Float] = [:]
    
        for (i, score) in confidences.enumerate() {
            classConfidences[Int(classLables[i])] = score
        }

        sensorDataCollection.setActivityTypePredictions(forClassConfidences: classConfidences)
        CoreDataManager.sharedManager.saveContext()
    }
}