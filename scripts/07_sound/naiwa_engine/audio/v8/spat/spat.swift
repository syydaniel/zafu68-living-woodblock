// Offline binaural renderer with Apple's own HRTF (AVAudioEnvironmentNode, .HRTFHQ): mono sources at 3D
// positions (moving if a path is given), distance attenuation and the environment reverb, rendered to a stereo wav for
// headphones. Usage: spat scene.json
// scene.json: { "out": "x.wav", "dur": seconds, "sr": 48000, "reverb": {"preset": "largeHall", "level": -12.0} | null,
//               "ref": refDistance m, "rolloff": 1.0, "maxdist": 200,
//               "src": [ {"file": "mono48k.wav", "t": start s, "gain": dB, "rv": reverbBlend 0-1,
//                         "pos": [[t, x, y, z], ...]  (x right, y up, z BEHIND the listener; front = -z) } ] }
import AVFoundation
import Foundation

struct Src: Decodable { let file: String; let t: Double; let gain: Double; let rv: Double?; let pos: [[Double]] }
struct Rv: Decodable { let preset: String; let level: Double }
struct Scene: Decodable { let out: String; let dur: Double; let sr: Double; let reverb: Rv?; let ref: Double?; let rolloff: Double?; let maxdist: Double?; let src: [Src] }

let args = CommandLine.arguments
guard args.count > 1, let data = FileManager.default.contents(atPath: args[1]) else { print("usage: spat scene.json"); exit(1) }
let scene = try! JSONDecoder().decode(Scene.self, from: data)
let sr = scene.sr
let engine = AVAudioEngine()
let env = AVAudioEnvironmentNode()
engine.attach(env)
let stereo = AVAudioFormat(standardFormatWithSampleRate: sr, channels: 2)!
let mono = AVAudioFormat(standardFormatWithSampleRate: sr, channels: 1)!
engine.connect(env, to: engine.mainMixerNode, format: stereo)
try! engine.enableManualRenderingMode(.offline, format: stereo, maximumFrameCount: 1024)
env.outputType = .headphones
env.listenerPosition = AVAudio3DPoint(x: 0, y: 0, z: 0)
env.listenerAngularOrientation = AVAudio3DAngularOrientation(yaw: 0, pitch: 0, roll: 0)
env.distanceAttenuationParameters.distanceAttenuationModel = .inverse
env.distanceAttenuationParameters.referenceDistance = Float(scene.ref ?? 1.0)
env.distanceAttenuationParameters.rolloffFactor = Float(scene.rolloff ?? 1.0)
env.distanceAttenuationParameters.maximumDistance = Float(scene.maxdist ?? 200.0)
if let rv = scene.reverb {
    let presets: [String: AVAudioUnitReverbPreset] = ["smallRoom": .smallRoom, "mediumRoom": .mediumRoom, "largeRoom": .largeRoom, "mediumHall": .mediumHall, "largeHall": .largeHall,
        "plate": .plate, "mediumChamber": .mediumChamber, "largeChamber": .largeChamber, "cathedral": .cathedral, "largeRoom2": .largeRoom2, "mediumHall2": .mediumHall2,
        "mediumHall3": .mediumHall3, "largeHall2": .largeHall2]
    env.reverbParameters.enable = true
    env.reverbParameters.loadFactoryReverbPreset(presets[rv.preset] ?? .largeHall)
    env.reverbParameters.level = Float(rv.level)
}
// voices: one player per source (attached up front; offline, so the count only costs time)
var players: [AVAudioPlayerNode] = []
func posAt(_ s: Src, _ t: Double) -> AVAudio3DPoint {
    let p = s.pos
    if p.count == 1 || t <= p[0][0] { return AVAudio3DPoint(x: Float(p[0][1]), y: Float(p[0][2]), z: Float(p[0][3])) }
    for i in 1..<p.count where t <= p[i][0] {
        let a = p[i - 1], b = p[i]; let u = (t - a[0]) / max(1e-6, b[0] - a[0])
        return AVAudio3DPoint(x: Float(a[1] + (b[1] - a[1]) * u), y: Float(a[2] + (b[2] - a[2]) * u), z: Float(a[3] + (b[3] - a[3]) * u))
    }
    let l = p[p.count - 1]; return AVAudio3DPoint(x: Float(l[1]), y: Float(l[2]), z: Float(l[3]))
}
var ends: [Double] = []
for s in scene.src {
    let f = try! AVAudioFile(forReading: URL(fileURLWithPath: s.file))
    let buf = AVAudioPCMBuffer(pcmFormat: f.processingFormat, frameCapacity: AVAudioFrameCount(f.length))!
    try! f.read(into: buf)
    // to mono float at the scene rate (the files are prepared as mono 48 k; keep channel 0 otherwise)
    let mb = AVAudioPCMBuffer(pcmFormat: mono, frameCapacity: buf.frameLength)!; mb.frameLength = buf.frameLength
    let g = Float(pow(10.0, s.gain / 20.0)); let src = buf.floatChannelData![0]; let dst = mb.floatChannelData![0]
    for i in 0..<Int(buf.frameLength) { dst[i] = src[i] * g }
    let p = AVAudioPlayerNode(); engine.attach(p); engine.connect(p, to: env, format: mono)
    p.renderingAlgorithm = .HRTFHQ; p.sourceMode = .spatializeIfMono
    p.reverbBlend = Float(s.rv ?? 0.0); p.position = posAt(s, s.t)
    players.append(p); ends.append(s.t + Double(buf.frameLength) / sr)
    p.scheduleBuffer(mb, at: AVAudioTime(sampleTime: AVAudioFramePosition(s.t * sr), atRate: sr), options: [], completionHandler: nil)
}
try! engine.start()
for p in players { p.play(at: AVAudioTime(sampleTime: 0, atRate: sr)) }
let total = AVAudioFramePosition(scene.dur * sr)
var outFile: AVAudioFile? = try! AVAudioFile(forWriting: URL(fileURLWithPath: scene.out), settings: [AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: sr, AVNumberOfChannelsKey: 2,
    AVLinearPCMBitDepthKey: 32, AVLinearPCMIsFloatKey: true, AVLinearPCMIsBigEndianKey: false], commonFormat: .pcmFormatFloat32, interleaved: false)
let rbuf = AVAudioPCMBuffer(pcmFormat: engine.manualRenderingFormat, frameCapacity: engine.manualRenderingMaximumFrameCount)!
while engine.manualRenderingSampleTime < total {
    let t = Double(engine.manualRenderingSampleTime) / sr
    for (i, s) in scene.src.enumerated() where s.pos.count > 1 && t >= s.t - 0.05 && t <= ends[i] + 0.05 { players[i].position = posAt(s, t) }
    let n = min(AVAudioFrameCount(total - engine.manualRenderingSampleTime), rbuf.frameCapacity)
    let st = try! engine.renderOffline(n, to: rbuf)
    if st == .success { try! outFile!.write(from: rbuf) } else { print("render status", st.rawValue); break }
}
engine.stop()
outFile = nil   // closes the file (the header is written on deinit)
print("SPAT DONE", scene.out, scene.src.count, "sources", scene.dur, "s")
