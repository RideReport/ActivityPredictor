//
//  RandomForest.swift
//  Ride
//
//  Created by William Henderson on 3/7/16.
//  Copyright © 2016 Knock Softwae, Inc. All rights reserved.
//

import Foundation

class RandomForestManager {
    var _ptr: COpaquePointer
    var classLables: [Int32]!
    var classCount = 0
    
    init(sampleSize: Int, samplingFrequency: Int) {
        let path = NSBundle(forClass: self.dynamicType).pathForResource("forest.cv", ofType: nil)
        let cpath = path?.cStringUsingEncoding(NSUTF8StringEncoding)

        _ptr = createRandomForestManager(Int32(sampleSize), Int32(samplingFrequency), UnsafeMutablePointer(cpath!))
        self.classCount = Int(randomForestGetClassCount(_ptr))
        self.classLables = [Int32](count:self.classCount, repeatedValue:0)
        randomForestGetClassLabels(_ptr, UnsafeMutablePointer(self.classLables), Int32(self.classCount))
    }
    
    deinit {
        deleteRandomForestManager(_ptr)
    }
    
    func classifyMagnitudeVector(accelerometerVector: [Float], gyroVector: [Float])->([Int: Float])
    {
        let confidences = [Float](count:self.classCount, repeatedValue:0.0)
        randomForestClassificationConfidences(_ptr, UnsafeMutablePointer(accelerometerVector), UnsafeMutablePointer(gyroVector), UnsafeMutablePointer(confidences), Int32(self.classCount))
        
        var classConfidences: [Int: Float] = [:]
    
        for (i, score) in confidences.enumerate() {
            classConfidences[Int(classLables[i])] = score
        }
        
        return classConfidences
    }
}