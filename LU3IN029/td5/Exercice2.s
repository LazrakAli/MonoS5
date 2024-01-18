
.data 
a:	.word	-5
b:	.word	3

.text
	lui	$8, 0x1001
	lw	$9, 0($8)
	lw	$10, 4($8)
#if (a != 0)
	beq	$9,$0,then
#a = a - b;
	subu	$9,$9,$10
	sw	$9,0($8)
	ori	$4,$9,0
	ori	$2,$0,1
	syscall
#exit()
	ori $2, $0,10
	syscall


then:	
	#a = a + b;
	add	$9, $9, $10
	sw	$9, 0($8)

	ori	$4,$9,0
	ori	$2,$0,1
	syscall
#exit()
	ori	$2, $0,10
	syscall