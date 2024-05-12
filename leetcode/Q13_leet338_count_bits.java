
class Q13_leet338_count_bits {
    public static int hammingWeight(int n) {
        int cnt = 0;
        int mask = 1;
        int sz = 0;
        while (sz < 32) {
            if ((n & mask) != 0) {
                cnt++;
            }
            mask = mask << 1;
            sz++;
        }
        return cnt;
    }

    public static int[] countBits(int n) {
        int[] bitcnt = new int[n+1];
        for (int i = 0; i <= n; i++) {
            bitcnt[i] = hammingWeight(i);
        } 
        return bitcnt;
    }

    public static void main(String[] args) {
        int n = 5;
        int[] x = countBits(n);
        for (int i = 0; i <= n; i++) {
            System.out.println(x[i]);
        }
    }
}
