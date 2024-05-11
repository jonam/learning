
class Q12_leet191_Number_1_bits {
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

    public static void main(String[] args) {
        System.out.println(hammingWeight(16));
    }
}
