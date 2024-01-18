.data

x:	.word	3
p:	.word	2

.text

main:
# prologue
	addiu	$29,$29,-16
	lui	$8,0x1001
# on recupere x et p et les enregistre dans la pile
	lw	$4,0($8)
	lw	$5,4($8)
	sw	$4,0($29)
	sw	$5,4($29)
#  printf("%d", puissance(x, p));
	jal	puissance
	ori	$4,$2,0
	ori	$2,$0,1
	syscall
	
#  printf("\n");
	ori	$4,$0,0x0A
	ori	$2,$0,11
	syscall 
	
# on enregistre 2 et 6 dans la pile
	ori	$4,$0,2
	sw	$4,8($29)
	ori	$5,$0,6
	sw	$5,12($29)
# printf("%d", puissance(2, 6));
	jal 	puissance
	ori	$4,$2,0
	ori	$2,$0,1
	syscall

# epilogue
	addiu	$29,$29,16
# exit
	ori	$2,$0,10
	syscall
	
puissance:
# prologue
	addiu	$29,$29,-16
	sw	$31,12($29)
	sw	$4,0($29)	
# if (n == 0)
	beq	$5,$0,n_0
# else
	addiu	$5,$5,-1
	sw	$5,4($29)
	jal	puissance
	mult	$2,$4
	mflo	$2
	j	fin

n_0:
	ori	$2,$0,1
fin:
	lw	$31,12($29)
	addiu	$29,$29,16
	jr	$31	
	
	
	
	
	
	
	
	
