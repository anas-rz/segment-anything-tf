from tensorflow import keras
import tensorflow.keras.backend as K

from tensorflow.keras import layers


class PatchEmbed(layers.Layer):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size = (16, 16),
        strides = (16, 16),
        padding= (0, 0),
        embed_dim = 768,
    ):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.pad = layers.ZeroPadding2D(padding=padding)
        self.proj = layers.Conv2D(
            embed_dim, kernel_size, strides=strides )

    def call(self, x):
        x = self.pad(x)
        print(x.shape)
        x = self.proj(x)
        # B C H W -> B H W C
        return x
    

class Attention(layers.Layer):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        num_heads =8,
        qkv_bias = True,
        # use_rel_pos = False,
        # rel_pos_zero_init = True,
        # input_size = None 
        ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias

    def build(self, input_shape):
        dim = input_shape[-1]
        head_dim = dim // self.num_heads
        self.scale = head_dim**-0.5

        self.qkv = layers.Dense(dim * 3, use_bias=self.qkv_bias)
        self.proj = layers.Dense(dim)

        print("Needs to implement rel_pos_bias")

    def call(self, x):
        B, H, W = K.int_shape(x)[0], K.int_shape(x)[1], K.int_shape(x)[2]
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, (B, H * W, 3, self.num_heads, -1))
        
        qkv = tf.transpose(qkv, perm=(2, 0, 3, 1, 4))
        # q, k, v with shape (B * nHead, H * W, C)
        qkv = tf.reshape(qkv, (3, B * self.num_heads, H * W, -1))
        
        q, k, v = tf.split(qkv, 3)
        k  = tf.transpose(k, perm=(0, 1, 3, 2))
        attn = (q * self.scale) @ k


        attn = tf.nn.softmax(attn, axis=-1)
        x = (attn @ v)
        x = tf.reshape(x, (B, self.num_heads, H, W, -1))
        x = tf.transpose(x, perm=(0, 2, 3, 1, 4))
        x = tf.reshape(x, shape=(B, H, W, -1))
        x = self.proj(x)

        return x

    

if __name__ == '__main__':
    import tensorflow as tf
    print("Testing Patch Embed")
    input_tensor = tf.random.uniform(shape=(1, 224, 224, 3))
    print(f"Input Shape {input_tensor.shape}")
    layer = PatchEmbed()
    out = layer(input_tensor)
    print(f"Output Shape {out.shape}")
    print("Testing Attention Embed")
    input_tensor = tf.random.uniform(shape=(1, 14, 14, 196))
    print(f"Input Shape {input_tensor.shape}")
    attn_layer = Attention(num_heads=14)
    out = attn_layer(input_tensor)
    print(f"Output Shape {out.shape}")



    