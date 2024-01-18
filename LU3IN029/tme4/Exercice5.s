.data

tab:	.word	4, 23, 12, 3, 8, 1 # 24 octets
s:	.space	4 #24(0x1001)
p:	.space	4 #28(0x1001)

.text

main:
	lui	$8,0x1001
#s = tab[3];
	lw	$9,12($8)
	sw	$9,20($8)
#p = tab[4];
	lw	$10,16($8)
	sw	$10, 28($8)
#tab[0] = s + 1;
	lw	$13, 24($8)
	lw	$11,0($8)
	addi	$11,$9,1
	sw	$11,0($8)
#tab[1] = s + p;
	lw	$14,28($8)
	lw	$12,4($8)
	add	$12,$9,$10
	sw	$11,4($8)
#tab[2] = tab[5];
	lw	$10, 24($8)
	sw	$10, 8($8)

#exit()
	ori $2,$0,10
	syscall