import tensorflow as tf

class Autoencoder():
    def __init__(self,list_enc_shapes =[], list_dec_shapes=[] ):
        self.encoders = [tf.keras.layers.Dense(x, activation='sigmoid') for x in list_enc_shapes]
        self.decoders = [tf.keras.layers.Dense(x, activation='sigmoid') for x in list_dec_shapes]

        self.model = tf.keras.Sequential(self.encoders + self.decoders)
        self.encoder = tf.keras.Sequential(self.encoders)
        self.decoder = tf.keras.Sequential(self.decoders)
    def generate_autoencoder(self):
        self.model.compile(optimizer='adam',
                      loss="mse")
        return self.encoder , self.decoder
    def fit_autoencoder(self,data,labels,val_data,val_labels,epochs,batchsize):
        return self.model.fit(data, labels, epochs=epochs, batch_size=batchsize, shuffle=True, validation_data=(val_data, val_labels))

    def encode(self,data):
        return self.encoder.predict(data)

    def decode(self,data):
        return self.decoder.predict(data)
    def save(self,name):
        self.model.save_weights(name+".h5f")
    def load(self,name):
        self.model.load_weights(name+".h5f")
        self.encoder = tf.keras.Sequential(self.encoders)
        self.decoder = tf.keras.Sequential(self.decoders)